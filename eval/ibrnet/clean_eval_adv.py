import sys
sys.path.append('../../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage, rng
from ibrnet.render_image import render_single_image
from ibrnet.render_ray import render_rays
from ibrnet.criterion import Criterion
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import tensorflow as tf
from lpips_tensorflow import lpips_tf
from torch.utils.data import DataLoader
import numpy as np

from train import calc_depth_var

import torch
import torch.nn as nn

from geo_interp import interp3
from pc_grad import PCGrad


# Clamping operation to make sure that the images are within pixel bounds
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# Computes a smoothness loss that encourages neighboring pixels in a depth map to have similar values, which results in smoother depth maps.
def calc_depth_smooth_loss(ret, patch_size, loss_type='l2'):
    depth = ret['depth']  # [N_rays,], i.e., [n_patches * patch_size * patch_size]
    
    depth = depth.reshape([-1, patch_size, patch_size])  # [n_patches, patch_size, patch_size]
    
    v00 = depth[:, :-1, :-1]
    v01 = depth[:, :-1, 1:]
    v10 = depth[:, 1:, :-1]

    if loss_type == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif loss_type == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported loss type.')
                
    return loss.sum()

# Class is computing Smooth L1 Loss(Huber Loss) for depth prediction tasks
class SL1Loss(nn.Module):
    # Construction Method, indicate loss function
    def __init__(self):
        super(SL1Loss, self).__init__()
        # Built-in PyTorch method, the average of the loss values will be returned
        self.loss = nn.SmoothL1Loss(reduction='mean')
    
    # Defines how the loss is computed when the class is called with input tensors
    # Returns: the computer loss value
    def forward(self, depth_pred, depth_gt, mask=None, useMask=True):
        if None == mask and useMask:
            # Only pixls where the ground truth depth is greater than 0 will be considered in the loss function
            mask = depth_gt > 0
        # Loss is computed here on the masked predicted depth and ground truth depth
        loss = self.loss(depth_pred[mask], depth_gt[mask])  # * 2 ** (1 - 2)
        return loss


# Performs a 3D to 2D projection between 2 different camera views
# using depth information from a reference camera. 
# Reprojectes points from the reference view to the source view, 
# computing their new coordinates and depths in the source view based on the camera intrinsics and extrinsic. 
# Part of 3D reconstruction methods, non-adversarial related. 
def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):  # (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')
    ''''''
    ## depth_ref: [B, H, W]
    ## intrinsics_ref: [3, 3]
    ## extrinsics_ref: [4, 4]
    ''''''
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                                   torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    pts = torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(
        0) * (depth_ref.view(batchsize, -1).unsqueeze(1))   ## [B, 3, height * width]

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), pts)   ## [B, 3, height * width]

    ### torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1).shape: [B, 4, height * width]
    ### torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)).shape: [4, 4]
    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                           torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:, :3, :]  ## [B, 3, height * width]

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480, [B, 3, height * width]
    depth_src = K_xyz_src[:, 2:3, :]   ## [B, 1, height * width]
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:, 2:3, :] + 1e-9)   ## [B, 2, height * width]
    x_src = xy_src[:, 0, :].view([batchsize, height, width])   ## [B, height, width]
    y_src = xy_src[:, 1, :].view([batchsize, height, width])   ## [B, height, width]

    return x_src, y_src, depth_src

# Performs a forward wraping operation from a reference camera view to a source camera view
# Forward warping is the process of projecting 3D points from one camera view to another 
# and then rendering them onto a 2D image plane in the target view
# Goal: make depth and RGB information frmo a reference view and project it to a source view, producing a new image and associated depth information
# Part of 3D reconstruction methods, non-adversarial related
def forward_warp(selected_inds, rgb_ref, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src, src2tar=True, derive_full_image=False, cpu_speedup=True):
    ''''''
    ## selected_inds: [Num_Sampled_Rays] from the target view
    ## rgb_ref: [H, W, C]
    ## depth_ref: [B, H, W] or [H, W]
    ## intrinsics_ref: [3, 3]
    ## extrinsics_ref: [4, 4]
    ## cpu_speedup: Put the indexing operations to CPU for further speed-up
    ''''''
    
    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]
    
    if cpu_speedup:
        new = torch.zeros([height, width, 3])  ## pseudo RGB for the image, [height, width, 3]
        # new = torch.zeros_like(rgb_ref)  ## deprecated, as rgb_ref may not have the same resolution as depth_ref
        
        depth_src = depth_src.reshape(height, width).cpu() ## [height, width], assume batch_size=1
        new_depth = torch.zeros_like(depth_src)  ## [height, width], depth for the whole source image
        x_res = x_res.cpu()
        y_res = y_res.cpu()
        yy_base, xx_base = torch.meshgrid([torch.arange(
            0, height, dtype=torch.long), torch.arange(0, width, dtype=torch.long)])
        
    else:
        new = torch.zeros([height, width, 3]).to(depth_ref.device)  ## pseudo RGB for the image, [height, width, 3]
        # new = torch.zeros_like(rgb_ref)  ## deprecated, as rgb_ref may not have the same resolution as depth_ref
        
        depth_src = depth_src.reshape(height, width) ## [height, width], assume batch_size=1
        new_depth = torch.zeros_like(depth_src)  ## [height, width], depth for the whole source image
        
        yy_base, xx_base = torch.meshgrid([torch.arange(
            0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long, device=depth_ref.device)])
    
    
    y_res = torch.clamp(y_res, 0, height - 1).to(torch.long)  ## [B, height, width]
    x_res = torch.clamp(x_res, 0, width - 1).to(torch.long)  ## [B, height, width]
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    y_res = y_res.reshape(-1)  ## [height*width], assume batch_size=1
    x_res = x_res.reshape(-1)  ## [height*width], assume batch_size=1
        
    if not derive_full_image:
        if src2tar:
            inds_res = (y_res * width + x_res).cpu().numpy()
            
            for i, inds in enumerate(inds_res):
                if inds in selected_inds:
                    if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                        new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                        new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]

            depth_proj = new_depth.reshape(-1)[selected_inds]  ## [N_rand]
            rgb_proj = new.reshape(-1,3)[selected_inds]   ## [N_rand, 3]
            
            if cpu_speedup:
                new = new.to(depth_ref.device)
                new_depth = new_depth.to(depth_ref.device)
                rgb_proj = rgb_proj.to(depth_ref.device)
                depth_proj = depth_proj.to(depth_ref.device)
                    
            return new, new_depth, rgb_proj, depth_proj
        
        else:
            selected_inds_new = []
            for i in selected_inds:
                selected_inds_new.append((y_res[i]*width + x_res[i]).item())
                if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                    new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                    new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]
            
            depth_proj = new_depth.reshape(-1)[selected_inds_new]  ## [N_rand]
            rgb_proj = new.reshape(-1,3)[selected_inds_new]   ## [N_rand, 3]

            if cpu_speedup:
                new = new.to(depth_ref.device)
                new_depth = new_depth.to(depth_ref.device)
                rgb_proj = rgb_proj.to(depth_ref.device)
                depth_proj = depth_proj.to(depth_ref.device)
                
            return new, new_depth, rgb_proj, depth_proj, selected_inds_new
        
    else:
        # painter's algo
        for i in range(yy_base.shape[0]):
            if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]

        depth_proj = new_depth.reshape(-1)[selected_inds]  ## [N_rand]
        rgb_proj = new.reshape(-1,3)[selected_inds]   ## [N_rand, 3]

        if cpu_speedup:
            new = new.to(depth_ref.device)
            new_depth = new_depth.to(depth_ref.device)
            rgb_proj = rgb_proj.to(depth_ref.device)
            depth_proj = depth_proj.to(depth_ref.device)
                
        return new, new_depth, rgb_proj, depth_proj


# Computes a 3D rotaion matrix given the input rotation angles anround the x,y,and z axes
# The rotation matrix then can be used to rotate a point or object in 3D space
def calc_rotation_matrix(rot_degree):
    # input: rot_degree - [3]
    # output: rotation matrix - [3, 3]
    
    dx, dy, dz = rot_degree
    
    rot_x = torch.zeros(3,3)
    rot_x[0,0]=torch.cos(dx)
    rot_x[0,1]=-torch.sin(dx)
    rot_x[1,0]=torch.sin(dx)
    rot_x[1,1]=torch.cos(dx)
    rot_x[2,2]=1
    
    rot_y = torch.zeros(3,3)
    rot_y[0,0]=torch.cos(dy)
    rot_y[0,2]=torch.sin(dy)
    rot_y[1,1]=1
    rot_y[2,0]=-torch.sin(dy)
    rot_y[2,2]=torch.cos(dy)

    rot_z = torch.zeros(3,3)
    rot_z[0,0]=1
    rot_z[1,1]=torch.cos(dz)
    rot_z[1,2]=-torch.sin(dz)
    rot_z[2,1]=torch.sin(dz)
    rot_z[2,2]=torch.cos(dz)

    rot_mat = rot_z.mm(rot_y.mm(rot_x))  # shape=[3, 3]
    
    return rot_mat

# Applied transformations(Rotation and translation) to a set of source camera poses using 
# given rotation and translation parameters. 
# Modifies camera's rotation matrix and translation vector for each source view
def transform_src_cameras(src_cameras, rot_param, trans_param, num_source_views):
    camera_pose = src_cameras[0, :, -16:].reshape(-1, 4, 4)  # [num_source_views, 4, 4]

    rot_mats = []
    for src_id in range(num_source_views):
        rot_mat = calc_rotation_matrix(rot_param[src_id])  # [3, 3]
        rot_mats.append(rot_mat.unsqueeze(0))
    rot_mats = torch.cat(rot_mats, dim=0).to(src_cameras)  # [num_source_views, 3, 3]

    rot_new = rot_mats.bmm(camera_pose[:, :3, :3])  # [num_source_views, 3, 3]       
    trans_new = camera_pose[:, :3, 3] + trans_param.to(src_cameras)  # [num_source_views, 3]
    rot_trans = torch.cat([rot_new, trans_new.unsqueeze(2)], dim=2)  # [num_source_views, 3, 4] 

    return rot_trans

"""MODIFY RANGE"""
# Intializes an adversarial perturbation to be applied to a batch of source rays 
# Input: A dictionary containing ray data, specifying RGB pixel values of source views
def init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit):
    # N_views is the number of source views 
    # H and W are the height and width of the image
    # 3 represents the RGB color channels
    delta = torch.zeros_like(src_ray_batch['src_rgbs'])  # ray_batch['src_rgbs'].shape=[1, N_views, H, W, 3]

    # Sample from a uniform distribution in the range [-eps, +eps]
    # Each element in delta = each RGB pixel value for each view
    # Each element is initialized with a random value between [-eps, +eps]
    delta.uniform_(-epsilon, epsilon)

    # Represents the upper and lower bounds of allowed RGB values 
    # Making sure the pixel values are still valid after perturbation
    delta.data = clamp(delta, lower_limit - src_ray_batch['src_rgbs'], upper_limit - src_ray_batch['src_rgbs'])
    
    # Enbales automatic differentiation for delta
    delta.requires_grad = True
    
    # Return initial perturbation
    return delta



"""MODIFY RANGE"""
# Performs optimization of delta applied to a ray-based rendering system
# Uses a combination of different losses
# Handles the creation of synthetic data, rendering, and adversarial perturbation 
# by calculating gradients based on a variety of loss functions, such as RGB reconstruction loss, depth consistency loss, density loss, and multi-view consistency losses. 
def optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True):
    
    # Checks if a GT depth path is provided
    if args.gt_depth_path:
        # If it exists, GT depth will be loaded and used later in depth-related losses
        load_gt_depth = True
    else:
        # Otherwise, the flag is set to be false. 
        load_gt_depth = False
    
    # Initialize Ray Sampler
    # Responsible for generating rays(light paths) to be rendered in the scene. 
    # The sampler can optionally load GT if available
    ray_sampler_train = RaySamplerSingleImage(data, 'cuda', load_gt_depth=load_gt_depth)     
    
    # Sample Rays for Training
    # Decides whether to use path sampling or random sampling
    if args.use_patch_sampling:
        # Patch sampling: sampling rays from a specific path
        # A specific number of rays are selected from patches of size
        train_ray_batch = ray_sampler_train.random_patch_sample(args.N_rand, patch_size=args.patch_size)
    else:
        # Random sampling, rays are randomly sampled from the whole image
        train_ray_batch = ray_sampler_train.random_sample(args.N_rand, sample_mode=args.sample_mode, center_ratio=args.center_ratio)

    # Optinal: Compute Pseudo-GT
    if args.use_pseudo_gt:

        # First generates a pseudo-GT RGB and depth map from the source rays
        with torch.no_grad():

            # Extracts feature maps
            featmaps = model.feature_net(src_ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))  ## extract features from test src_rgbs
            
            # Render rays, contains fine RGB and depth outputs('outputs_fine')
            # "ret_gt" is then used as "pseudo-GT" to guide the optimization
            ret_gt = render_rays(ray_batch=train_ray_batch,
                            model=model,
                            projector=projector,
                            featmaps=featmaps,
                            N_samples=args.N_samples,
                            inv_uniform=args.inv_uniform,
                            N_importance=args.N_importance,
                            det=True, # args.det,
                            white_bkgd=args.white_bkgd,
                            args=args,
                            src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch
            train_ray_batch['rgb'] = ret_gt['outputs_fine']['rgb']
            train_ray_batch['depth'] = ret_gt['outputs_fine']['depth']
    
    """ALL LOSS FUNCTIONS ARE HERE"""


    """------Calculating loss in terms of RGB, Density, Depth Var, Depth Diff, Depth Consistency, Depth Smoothness, Camera Consistency------"""
    
    total_loss = {}
    
    # Delta is added to the source RGB images
    # The modified source images are passed through the "feature_net" to extract feature maps
    # These feature maps are used to render the adversarially perturbed rays
    featmaps = model.feature_net((src_ray_batch['src_rgbs']+delta).squeeze(0).permute(0, 3, 1, 2))  ## extract features from test src_rgbs
    
    # Takes the sampled rays, the model, and the perturbed feature maps to render the scene
    # The result "ret" contains the rendered RGB and depth values of both coarse and fine outputs
    ret = render_rays(ray_batch=train_ray_batch,
                    model=model,
                    projector=projector,
                    featmaps=featmaps,
                    N_samples=args.N_samples,
                    inv_uniform=args.inv_uniform,
                    N_importance=args.N_importance,
                    det=args.det,
                    white_bkgd=args.white_bkgd,
                    args=args,
                    src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch

    """------Mandatory Calculate RGB Loss(criterion)------"""

    # The RGB reconstruction loss "loss_rgb" is calculated between the rendered outputs and the GT RGB values
    loss_rgb, _ = criterion(ret['outputs_coarse'], train_ray_batch, scalars_to_log=None)
    
    # If fine outputs are available, an additional loss is computed and added to the total RGB loss
    if ret['outputs_fine'] is not None:
        fine_loss, _ = criterion(ret['outputs_fine'], train_ray_batch, scalars_to_log=None)
        loss_rgb += fine_loss  
    total_loss['rgb'] = loss_rgb

    """------Optional Density Loss Computation(MSE)------"""

    # If density loss is enabled, it calculates the mean squared error between the predicted alpha values(opacity) and the GT alpha values 
    # Ensure that the model predicts consistent transparency levels across views
    if args.density_loss > 0:
        assert args.use_pseudo_gt
        from utils import img2mse
        loss_density = img2mse(ret['outputs_coarse']['alpha'], ret_gt['outputs_coarse']['alpha'])

        if ret['outputs_fine'] is not None:
            loss_density += img2mse(ret['outputs_fine']['alpha'], ret_gt['outputs_fine']['alpha'])
        
        loss_density = args.density_loss * loss_density
        total_loss['density'] = loss_density
    
    """------Optional Depth Variance Loss------"""

    # Penalizes inconsistency in depth values across rays
    # Encourages smoother depth predictions by minimizing the variance of the depth map
    if args.depth_var_loss > 0:
        depth_var = calc_depth_var(ret['outputs_coarse'])
        
        if ret['outputs_fine'] is not None:
            depth_var += calc_depth_var(ret['outputs_fine']) 
                
        loss_depth_var = args.depth_var_loss * depth_var
        total_loss['depth_var'] = loss_depth_var
        
    """------Optional Difference Loss(SL1 Loss)------"""

    # Computed using Smooth L1 Loss between the predicted depth map and the GT depth map
    # Encourages the predicted depth to closely match the GT depth
    if args.depth_diff_loss > 0:
        sl1_loss = SL1Loss()
        depth_gt = train_ray_batch['depth']
        depth_pred = ret['outputs_coarse']['depth']
        
        depth_diff = sl1_loss(depth_pred, depth_gt, useMask=True)
        if ret['outputs_fine'] is not None:
            depth_pred = ret['outputs_fine']['depth']
            depth_diff += sl1_loss(depth_pred, depth_gt, useMask=True)
            
        loss_depth_diff = args.depth_diff_loss * depth_diff
        total_loss['depth_diff'] = loss_depth_diff
        
    """------Optional Depth Consistency Loss (SL1 Loss)------"""

    # Enforces consistency between the predicted depth maps of different views
    # Ensures that multi-view depth predictions align with each other
    ### add multi-view depth consistency loss following Sin-NeRF
    if args.depth_consistency_loss > 0:
        sl1_loss = SL1Loss()

        if args.ds_rgb:  # downsample rgb images to match the resolution of depth; otherwise use upsampled depth
            ray_sampler_train_cons = RaySamplerSingleImage(data, 'cuda', resize_factor=0.5, load_gt_depth=True)  # to match the resolution of ground-truth depth     
                
            if args.use_patch_sampling:
                train_ray_batch_cons = ray_sampler_train_cons.random_patch_sample(args.N_rand, patch_size=args.patch_size)
            else:
                train_ray_batch_cons = ray_sampler_train_cons.random_sample(args.N_rand, sample_mode=args.sample_mode, center_ratio=args.center_ratio)
                
            featmaps = model.feature_net((src_ray_batch['src_rgbs']+delta).squeeze(0).permute(0, 3, 1, 2))  ## extract features from test src_rgbs
            
            ret_cons = render_rays(ray_batch=train_ray_batch_cons,
                            model=model,
                            projector=projector,
                            featmaps=featmaps,
                            N_samples=args.N_samples,
                            inv_uniform=args.inv_uniform,
                            N_importance=args.N_importance,
                            det=args.det,
                            white_bkgd=args.white_bkgd,
                            args=args,
                            src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch
        
        else:
            train_ray_batch_cons = train_ray_batch
            ret_cons = ret
    
        train_camera = train_ray_batch_cons['camera'] # [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        extrinsics_src = train_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_src = train_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_src = intrinsics_src[:3, :3] # [3, 3]
        
        src_id = rng.choice(args.num_source_views)
        src_cameras = src_ray_batch['src_cameras']  # [1, num_source_views, 34]
        src_camera = src_cameras[0, src_id:src_id+1, :]
        
        extrinsics_ref = src_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_ref = src_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_ref = intrinsics_ref[:3, :3] # [3, 3]
        
        src_rgbs = src_ray_batch['src_rgbs']  # [1, num_source_views, H, W, 3]
        rgb_ref = src_rgbs[:, src_id, :, :]   # [1, H, W, 3]

        if args.ds_rgb:
            ### resize the rgb and intrinsics of the ref view to match the depth resolution
            resize_factor = 0.5
            intrinsics_ref[:2, :3] *= resize_factor
            rgb_ref = torch.nn.functional.interpolate(rgb_ref.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)
        
        rgb_ref = rgb_ref[0]  # [H, W, 3]
        
        src_depths = src_ray_batch['src_depths']    # [1, num_source_views, H, W]
        depth_ref = src_depths[0, src_id:src_id+1, :, :]  # [1, H, W]
        
        rgb_proj_full, depth_proj_full, rgb_proj, depth_proj = forward_warp(train_ray_batch_cons['selected_inds'], rgb_ref, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src, src2tar=True, derive_full_image=False)
            
        ## Visualize the warpped depth and rgb
        # depth_src_colored = colorize_np(depth_ref[0].cpu().numpy(), range=tuple(data['depth_range'].squeeze().cpu().numpy()))
        # imageio.imwrite(os.path.join(out_scene_dir, 'depth_src_{}.png'.format(num_iter)),(255 * depth_src_colored).astype(np.uint8))

        # depth_target_colored = colorize_np(train_ray_batch_cons['depth_full'][0].cpu().numpy(), range=tuple(data['depth_range'].squeeze().cpu().numpy()))
        # imageio.imwrite(os.path.join(out_scene_dir, 'depth_target_{}.png'.format(num_iter)),(255 * depth_target_colored).astype(np.uint8))
        
        # depth_proj_full_colored = colorize_np(depth_proj_full.cpu().numpy(), range=tuple(data['depth_range'].squeeze().cpu().numpy()))
        # imageio.imwrite(os.path.join(out_scene_dir, 'depth_proj_{}.png'.format(num_iter)),(255 * depth_proj_full_colored).astype(np.uint8))

        # rgb_src_vis = (255 * np.clip(src_ray_batch['src_rgbs'][0][src_id].cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        # imageio.imwrite(os.path.join(out_scene_dir, 'rgb_src_{}.png'.format(num_iter)), rgb_src_vis)
        
        # rgb_full_vis = (255 * np.clip(data['rgb'][0].cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        # imageio.imwrite(os.path.join(out_scene_dir, 'rgb_target_{}.png'.format(num_iter)), rgb_full_vis)
        
        # rgb_proj_full_vis = (255 * np.clip(rgb_proj_full.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        # imageio.imwrite(os.path.join(out_scene_dir, 'rgb_proj_{}.png'.format(num_iter)), rgb_proj_full_vis)
        # print("Saved!!!")
        # input()
    
        depth_pred = ret_cons['outputs_coarse']['depth']
        loss_depth_cons = sl1_loss(depth_pred, depth_proj, useMask=True)
        if ret_cons['outputs_fine'] is not None:
            depth_pred = ret_cons['outputs_fine']['depth']
            loss_depth_cons += sl1_loss(depth_pred, depth_proj, useMask=True)
            
        loss_depth_cons = args.depth_consistency_loss * loss_depth_cons
        total_loss['depth_cons'] = loss_depth_cons

    """------Optional Depth Smoothness Loss (calc_depth_smooth_loss)------"""

    # Encourages smoothness in the predicted depth map by penalizing abrupt changes in depth values within a local patch
    # Similar to total variation loss...
    ### apply depth smooth loss following RegNeRF
    if args.depth_smooth_loss > 0:
        if not args.use_patch_sampling:
            train_ray_batch = ray_sampler_train.random_patch_sample(args.N_rand, patch_size=args.patch_size)

            featmaps = model.feature_net((src_ray_batch['src_rgbs']+delta).squeeze(0).permute(0, 3, 1, 2))  ## extract features from test src_rgbs

            ret_smooth = render_rays(ray_batch=train_ray_batch,
                            model=model,
                            projector=projector,
                            featmaps=featmaps,
                            N_samples=args.N_samples,
                            inv_uniform=args.inv_uniform,
                            N_importance=args.N_importance,
                            det=args.det,
                            white_bkgd=args.white_bkgd,
                            args=args,
                            src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch
        
        else:
            ret_smooth = ret

        loss_depth_smooth = calc_depth_smooth_loss(ret_smooth['outputs_coarse'], patch_size=args.patch_size)
        if ret_smooth['outputs_fine'] is not None:
            loss_depth_smooth += calc_depth_smooth_loss(ret_smooth['outputs_fine'], patch_size=args.patch_size)
        
        loss_depth_smooth = args.depth_smooth_loss * loss_depth_smooth
        total_loss['depth_smooth'] = loss_depth_smooth

    
    """------Optional Camera Consistency Loss(SL1 Loss)"""

    # Ensures that the RGB and depth predictions remain consistent when the camera pose is slightly perturbed
    # Enforces view consistency 
    ### add multi-view consistency loss for perturbing the camera
    if args.camera_consistency_loss > 0:
        sl1_loss = SL1Loss()
    
        train_camera = train_ray_batch['camera'] # [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        extrinsics_tar = train_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_tar = train_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_tar = intrinsics_tar[:3, :3] # [3, 3]
        
        src_id = rng.choice(args.num_source_views)
        src_cameras = src_ray_batch['src_cameras']  # [1, num_source_views, 34]
        src_camera = src_cameras[0, src_id:src_id+1, :]

        depth_tar = train_ray_batch['depth_full']  # [1, H, W]
        rgb_tar = ray_sampler_train.rgb.reshape(ray_sampler_train.H, ray_sampler_train.W, 3)  # [H, W, 3]
        
        extrinsics_src = src_camera[:, -16:].reshape(4, 4)  # [4, 4]
        intrinsics_src = src_camera[:, 2:18].reshape(4, 4)  # [4, 4]
        intrinsics_src = intrinsics_src[:3, :3] # [3, 3]
        
        src_rgbs = src_ray_batch['src_rgbs']  # [1, num_source_views, H, W, 3]
        rgb_src = src_rgbs[:, src_id, :, :]   # [1, H, W, 3]
        rgb_src = rgb_src[0]  # [H, W, 3]
        
        src_depths = src_ray_batch['src_depths']    # [1, num_source_views, H, W]
        depth_src = src_depths[0, src_id:src_id+1, :, :]  # [1, H, W]
    
        rgb_src2tar_full, depth_src2tar_full, rgb_src2tar, depth_src2tar = forward_warp(train_ray_batch['selected_inds'], rgb_src, depth_src, intrinsics_src, extrinsics_src, intrinsics_tar, extrinsics_tar, src2tar=True, derive_full_image=False)
        rgb_tar2src_full, depth_tar2src_full, rgb_tar2src, depth_tar2src, selected_inds_src = forward_warp(train_ray_batch['selected_inds'], rgb_tar, depth_tar, intrinsics_tar, extrinsics_tar, intrinsics_src, extrinsics_src, src2tar=False, derive_full_image=False)

        if args.perturb_camera_no_detach:
            rgb_tar_sampled = ret['outputs_fine']['rgb']  # [N_rand, 3]
        else:
            rgb_tar_sampled = ret['outputs_fine']['rgb'].detach()  # [N_rand, 3]
        depth_tar_sampled = train_ray_batch['depth']   # [N_rand]
        rgb_src_sampled = rgb_src.reshape(-1, 3)[selected_inds_src]  # [N_rand, 3]
        depth_src_sampled = depth_src.reshape(-1)[selected_inds_src]  # [N_rand]
        
        loss_camera_cons = args.cam_src2tar * sl1_loss(rgb_tar_sampled, rgb_src2tar, useMask=True) + args.cam_tar2src * sl1_loss(rgb_src_sampled, rgb_tar2src, useMask=True)
        loss_camera_cons += args.cam_depth * (sl1_loss(depth_tar_sampled, depth_src2tar, useMask=True) + sl1_loss(depth_src_sampled, depth_tar2src, useMask=True))

        loss_camera_cons = args.camera_consistency_loss * loss_camera_cons
        total_loss['camera_cons'] = loss_camera_cons

    """------ALL LOSSES CALCULATION DONE------"""

    # Sum up all losses we want to take into consideration
    loss = sum(total_loss.values())
    
    # Return both the sum of losses and individual loss here
    if return_loss:
        return loss, total_loss

    # Otherwise, it computes and returns the gradient of the total loss 
    # with respect to the adversarial perturbation delta, 
    # which can be used to update the perturbation during the optimization
    grad = torch.autograd.grad(loss, delta)[0].detach()
    
    return grad

    
def initialize_delta(args, train_loader, load_gt_depth):
    # NerFool+ code only. Ignore
    src_ray_batch = None
    if not args.view_specific and not args.no_attack:
        test_dataset_for_src = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes, use_glb_src=args.use_center_view)
        test_loader_for_src = DataLoader(test_dataset_for_src, batch_size=1)
        
        for i, data in enumerate(test_loader_for_src):   
            src_ray_sampler = RaySamplerSingleImage(data, device='cuda', load_gt_depth=load_gt_depth)
            src_ray_batch_glb = src_ray_sampler.get_all()  # global source views for all view directions
            break
    
    # NerFool or no attack 
    else:
        src_ray_batch_glb = None

    # No attack specified
    if args.no_attack:
        delta = 0
        print("No attack, Whether to Use View-specific Source Views:", args.view_specific)

    # NeRFool Attack 
    elif args.view_specific:
        print("Attack with View-specific Adv Perturbations")
    
    # NerFool+ Attack, Ignore
    else:   # optimize generalizable adv perturb across different views
        print("Attack with Adv Perturbations Generalizable across Views...")
        
        src_ray_batch = src_ray_batch_glb
        
        epsilon = torch.tensor(args.epsilon / 255.).cuda()
        alpha = torch.tensor(args.adv_lr / 255.).cuda()
        upper_limit = 1
        lower_limit = 0

        if args.perturb_camera:                
            rot_param = torch.zeros(args.num_source_views, 3)  ## 3 rotation degrees
            if not args.zero_camera_init:
                rot_param.uniform_(-args.rot_epsilon/180 * np.pi, args.rot_epsilon/180 * np.pi)
            rot_param.requires_grad = True
            
            trans_param = torch.zeros(args.num_source_views, 3)  ## 3 translation distances
            if not args.zero_camera_init:
                trans_param.uniform_(-args.trans_epsilon, args.trans_epsilon)
            trans_param.requires_grad = True

            src_cameras_orig = src_ray_batch['src_cameras'].clone()  # [1, num_source_views, 34]
                
        delta = init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit)

        if args.use_adam:
            if args.perturb_camera:
                params = [delta, rot_param, trans_param]
            else:
                params = [delta]
                    
            opt = torch.optim.Adam(params, lr=args.adam_lr)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)

            if args.use_pcgrad:
                opt = PCGrad(opt, num_source_views=args.num_source_views)
        
        print("Start adversarial perturbation...")
        
        continue_flag = True
        iters = 0
        while continue_flag:
            for data in train_loader:
                if args.use_unseen_views:
                    args.use_pseudo_gt = True   # as GT rgb/depth may not be available for unseen views
                    
                    if args.sample_based_on_depth:
                        z_camera = np.array([pose[2,2] for pose in train_dataset.render_poses])  # forward direction
                        p_camera = np.exp(z_camera/args.temp) / np.sum(np.exp(z_camera/args.temp))
                        poses_sample_id = np.random.choice(len(train_dataset.render_poses), size=3, p=p_camera, replace=False)
                    
                    else:
                        poses_sample_id = np.random.choice(len(train_dataset.render_poses), size=3, replace=False)
                    
                    pose1 = train_dataset.render_poses[poses_sample_id[0]]
                    pose2 = train_dataset.render_poses[poses_sample_id[1]]
                    pose3 = train_dataset.render_poses[poses_sample_id[2]]
                    
                    if args.decouple_interp_range:
                        s12_rot, s3_rot = np.random.uniform(0, args.interp_upbound_rot, size=2)
                        s12_trans, s3_trans = np.random.uniform(0, args.interp_upbound_trans, size=2)
                        s12 = [s12_rot, s12_trans]
                        s3 = [s3_rot, s3_trans]
                        
                    else:                 
                        if args.sample_based_on_depth:
                            s12, s3 = np.random.beta(args.beta, args.beta, size=2) * args.interp_upbound_rot
                        else:
                            s12, s3 = np.random.uniform(0, args.interp_upbound, size=2)
                                        
                    pose = interp3(pose1, pose2, pose3, s12, s3)  # [4, 4]
                    pose = pose.flatten().unsqueeze(0)  # [1, 16]
                         
                    camera_orig = data['camera'] # [1, 34]
                    pose = pose.to(camera_orig)
                    
                    camera_new = torch.cat([camera_orig[:,:18], pose], dim=1)
                    data['camera'] = camera_new
                    

                if args.perturb_camera:
                    rot_trans = transform_src_cameras(src_cameras_orig, rot_param, trans_param, args.num_source_views)  # [num_source_views, 3, 4]
                    rot_trans = rot_trans.reshape(-1, 12)  # [num_source_views, 12] 
                    src_ray_batch['src_cameras'] = torch.cat([src_cameras_orig[:,:,:-16], rot_trans.unsqueeze(0), src_cameras_orig[:,:,-4:]], dim=2)
                                        
                if args.use_adam:
                    loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True)
                    
                    opt.zero_grad()

                    if args.use_pcgrad:
                        opt.pc_backward(loss_dict, major_loss=args.major_loss)
                    else:
                        loss.backward()

                    delta.grad.data *= -1
                    
                    if args.perturb_camera and not args.perturb_camera_no_opt:
                        rot_param.grad.data *= -1
                        trans_param.grad.data *= -1
                        
                    opt.step()
                    scheduler.step()
                    
                else:  
                    loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True)
                    
                    loss.backward()
                    grad = delta.grad.detach()
                    if args.use_l2:
                        # Normalize the gradients to have unit L2 norm.
                        grad_norm = torch.norm(grad.view(len(delta), -1), p=2, dim=1).clamp(min=1e-20)
                        grad = grad / grad_norm.view(len(delta), 1, 1, 1)
    
                        perturbed_delta_data = delta.data.detach() + alpha * torch.sign(grad)

                        # Project perturbation delta to have L2 norm less than or equal to eps.
                        change_in_delta = perturbed_delta_data - delta.data.detach()
                        l2_delta_change = change_in_delta.renorm(p=2, dim=0, maxnorm=epsilon)
                        with torch.no_grad():
                            delta += l2_delta_change
                    else:
                        delta.data = delta.data + alpha * torch.sign(grad)
                    delta.grad.zero_()

                    if args.perturb_camera:
                        grad = rot_param.grad.detach()

                        if args.use_l2:
                            # Normalize the gradients to have unit L2 norm.
                            grad_norm = torch.norm(grad.view(len(rot_param), -1), p=2, dim=1).clamp(min=1e-20)
                            grad = grad / grad_norm.view(len(rot_param), 1, 1, 1)

                            perturbed_rot_param_data = rot_param.data.detach() + args.adv_lr * torch.sign(grad)
                        
                            # Project perturbation delta to have L2 norm less than or equal to eps.
                            change_in_delta = perturbed_rot_param_data - rot_param.data.detach()
                            l2_delta_change = change_in_delta.renorm(p=2, dim=0, maxnorm=args.rot_epsilon / 180 * np.pi)
                            with torch.no_grad(): rot_param.data = rot_param.data + l2_delta_change
                        else:
                            rot_param.data = rot_param.data + args.adv_lr * torch.sign(grad)

                        rot_param.grad.zero_()

                        grad = trans_param.grad.detach()
                        if args.use_l2:
                            # Normalize the gradients to have unit L2 norm.
                            grad_norm = torch.norm(grad.view(len(trans_param), -1), p=2, dim=1).clamp(min=1e-20)
                            grad = grad / grad_norm.view(len(trans_param), 1, 1, 1)

                            perturbed_trans_param_data = trans_param.data.detach() + args.adv_lr * torch.sign(grad)

                            # Project perturbation delta to have L2 norm less than or equal to eps.
                            change_in_delta = perturbed_trans_param_data - trans_param.data.detach()
                            l2_delta_change = change_in_delta.renorm(p=2, dim=0, maxnorm=args.trans_epsilon / 180 * np.pi)
                            with torch.no_grad(): trans_param.data = trans_param.data + l2_delta_change
                        else:
                            trans_param.data = trans_param.data + args.adv_lr * torch.sign(grad)
                        trans_param.grad.zero_()   
                        
                delta.data = clamp(delta.data, -epsilon, epsilon)
                delta.data = clamp(delta.data, lower_limit - src_ray_batch['src_rgbs'], upper_limit - src_ray_batch['src_rgbs'])  
                
                if args.perturb_camera:
                    rot_param.data = clamp(rot_param.data, torch.tensor(-args.rot_epsilon/180 * np.pi).cuda(), torch.tensor(args.rot_epsilon/180 * np.pi).cuda())
                    trans_param.data = clamp(trans_param.data, torch.tensor(-args.trans_epsilon).cuda(), torch.tensor(args.trans_epsilon).cuda())  
                    
                iters += 1
                if iters > args.adv_iters:
                    continue_flag = False
                    break

    return src_ray_batch


def render_and_evaluate(args, cur_index, model, data, delta, src_ray_batch, ray_sampler, ray_batch, file_id, results_dict, current_scene_results_dict, metrics):
    """--------------- Render Image and Evaluate ----------------"""
    with torch.no_grad():
        # ray_sampler = RaySamplerSingleImage(data, device='cuda', load_gt_depth=load_gt_depth)
        # ray_batch = ray_sampler.get_all()
        featmaps = model.feature_net((src_ray_batch['src_rgbs']+delta).squeeze(0).permute(0, 3, 1, 2))

        if args.use_clean_color or args.use_clean_density:
            featmaps_clean = model.feature_net((src_ray_batch['src_rgbs']).squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps_clean = None
                
        ret = render_single_image(ray_sampler=ray_sampler,
                                    ray_batch=ray_batch,
                                    model=model,
                                    projector=projector,
                                    chunk_size=args.chunk_size,
                                    det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=featmaps,
                                      args=args,
                                      featmaps_clean=featmaps_clean,
                                      src_ray_batch=src_ray_batch)
            
        gt_rgb = data['rgb'][0]
        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
        coarse_err_map = torch.sum((coarse_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
        coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_coarse.png'.format(file_id)), coarse_err_map_colored)
        coarse_pred_rgb_np = np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
        gt_rgb_np = gt_rgb.numpy()[None, ...]

        # different implementation of the ssim and psnr metrics can be different.
        # we use the tf implementation for evaluating ssim and psnr to match the setup of NeRF paper.

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session() as session:
            coarse_lpips = session.run(metrics['distance_t'], feed_dict={metrics['pred_ph']: coarse_pred_rgb_np, metrics['gt_ph']: gt_rgb_np})[0]
            coarse_ssim = session.run(metrics['ssim_tf'], feed_dict={metrics['pred_ph']: coarse_pred_rgb_np, metrics['gt_ph']: gt_rgb_np})[0]
            coarse_psnr = session.run(metrics['psnr_tf'], feed_dict={metrics['pred_ph']: coarse_pred_rgb_np, metrics['gt_ph']: gt_rgb_np})[0]

        # saving outputs ...
        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(file_id)), coarse_pred_rgb)

        gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_gt_rgb.png'.format(file_id)), gt_rgb_np_uint8)

        coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(file_id)),
                            (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                    range=tuple(data['depth_range'].squeeze().cpu().numpy()))
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(file_id)),
                            (255 * coarse_pred_depth_colored).astype(np.uint8))
        coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()
        coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(file_id)),
                            coarse_acc_map_colored)

        current_scene_results_dict['sum_coarse_psnr'] += coarse_psnr
        current_scene_results_dict['running_mean_coarse_psnr'] = current_scene_results_dict['sum_coarse_psnr'] / (cur_index + 1)
        current_scene_results_dict['sum_coarse_lpips'] += coarse_lpips
        current_scene_results_dict['running_mean_coarse_lpips'] = current_scene_results_dict['sum_coarse_lpips'] / (cur_index + 1)
        current_scene_results_dict['sum_coarse_ssim'] += coarse_ssim
        current_scene_results_dict['running_mean_coarse_ssim'] = current_scene_results_dict['sum_coarse_ssim'] / (cur_index + 1)

        if ret['outputs_fine'] is not None:
            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)

            with tf.Session() as session:
                fine_lpips = session.run(metrics['distance_t'], feed_dict={metrics['pred_ph']: fine_pred_rgb_np, metrics['gt_ph']: gt_rgb_np})[0]
                fine_ssim = session.run(metrics['ssim_tf'], feed_dict={metrics['pred_ph']: fine_pred_rgb_np,metrics['gt_ph']: gt_rgb_np})[0]
                fine_psnr = session.run(metrics['psnr_tf'], feed_dict={metrics['pred_ph']: fine_pred_rgb_np, metrics['gt_ph']: gt_rgb_np})[0]

            fine_err_map = torch.sum((fine_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
            fine_err_map_colored = (colorize_np(fine_err_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_fine.png'.format(file_id)),
                                fine_err_map_colored)

            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(file_id)), fine_pred_rgb)
            fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(file_id)),
                                (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
                
            fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                      range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(file_id)),
                                (255 * fine_pred_depth_colored).astype(np.uint8))

            if 'depth' in data:
                fine_gt_depth_colored = colorize_np(data['depth'][0],
                                                    range=tuple(data['depth_range'].squeeze().cpu().numpy()))
                imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_GT.png'.format(file_id)),
                                (255 * fine_gt_depth_colored).astype(np.uint8))
                
            fine_acc_map = torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()
            fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(file_id)),
                            fine_acc_map_colored)
        else:
            fine_ssim = fine_lpips = fine_psnr = 0.

        current_scene_results_dict['sum_fine_psnr'] += fine_psnr
        current_scene_results_dict['running_mean_fine_psnr'] = current_scene_results_dict['sum_fine_psnr'] / (cur_index + 1)
        current_scene_results_dict['sum_fine_lpips'] += fine_lpips #[0]
        #print(current_scene_results_dict['sum_fine_lpips'], cur_index)
        current_scene_results_dict['running_mean_fine_lpips'] = current_scene_results_dict['sum_fine_lpips'] / (cur_index + 1)
        current_scene_results_dict['sum_fine_ssim'] += fine_ssim
        current_scene_results_dict['running_mean_fine_ssim'] = current_scene_results_dict['sum_fine_ssim'] / (cur_index + 1)

        print(current_scene_results_dict['running_mean_fine_lpips'])
        print("==================\n"
                  "{}, curr_id: {} \n"
                  "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                  "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                  "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                  "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                  "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                  "running mean coarse lpips: {:03f}, running mean fine lpips: " + str(current_scene_results_dict['running_mean_fine_lpips']) + 
                  "===================\n"
                  .format(scene_name, file_id,
                          coarse_psnr, fine_psnr,
                          current_scene_results_dict['running_mean_coarse_psnr'],
                          current_scene_results_dict['running_mean_fine_psnr'],
                          coarse_ssim, fine_ssim,
                          current_scene_results_dict['running_mean_coarse_ssim'], 
                          current_scene_results_dict['running_mean_fine_ssim'],
                          coarse_lpips, fine_lpips,
                          current_scene_results_dict['running_mean_coarse_lpips']
                          #, 
                          #current_scene_results_dict['running_mean_fine_lpips']
                ))  # {:03f} \n"

        results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                                'fine_psnr': fine_psnr,
                                                'coarse_ssim': coarse_ssim,
                                                'fine_ssim': fine_ssim,
                                                'coarse_lpips': coarse_lpips,
                                                'fine_lpips': fine_lpips,
                                            }

    mean_coarse_psnr = current_scene_results_dict['sum_coarse_psnr'] / total_num
    mean_fine_psnr = current_scene_results_dict['sum_fine_psnr'] / total_num
    mean_coarse_lpips = current_scene_results_dict['sum_coarse_lpips'] / total_num
    mean_fine_lpips = current_scene_results_dict['sum_fine_lpips'] / total_num
    mean_coarse_ssim = current_scene_results_dict['sum_coarse_ssim'] / total_num
    mean_fine_ssim = current_scene_results_dict['sum_fine_ssim'] / total_num

    print('------{}-------\n'
          'final coarse psnr: {}, final fine psnr: {}\n'
          'fine coarse ssim: {}, final fine ssim: {} \n'
          'final coarse lpips: {}, fine fine lpips: {} \n'
          .format(scene_name, mean_coarse_psnr, mean_fine_psnr,
                  mean_coarse_ssim, mean_fine_ssim,
                  mean_coarse_lpips, mean_fine_lpips,
                  ))

    results_dict[scene_name]['coarse_mean_psnr'] = mean_coarse_psnr
    results_dict[scene_name]['fine_mean_psnr'] = mean_fine_psnr
    results_dict[scene_name]['coarse_mean_ssim'] = mean_coarse_ssim
    results_dict[scene_name]['fine_mean_ssim'] = mean_fine_ssim
    results_dict[scene_name]['coarse_mean_lpips'] = mean_coarse_lpips
    results_dict[scene_name]['fine_mean_lpips'] = mean_fine_lpips

    f = open("{}/psnr_{}_{}.txt".format(extra_out_dir, save_prefix, model.start_step), "w")
    f.write(str(results_dict))
    f.close()

    
def Run_NerFool(args, model, test_loader, src_ray_batch_glb, out_scene_dir, load_gt_depth, results_dict, current_scene_results_dict, metrics):
    # Loop over test data
    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0] # Retrive path for image 
        file_id = os.path.basename(rgb_path).split('.')[0] # Extract unique identifier for the image
        src_rgbs = data['src_rgbs'][0].cpu().numpy() # Convert source RGB images from Pytorch tensor to Numpy array for processing

        # Compute the averaged image: 
        # Takes the mean of the source RGB images across all views, 
        # multiplies it by 255 to convert it to standard RGB pixel values
        # cast it to "uint8"
        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)),
                        averaged_img)

        # Swich model to eval mode
        model.switch_to_eval()
        
        # Ray sampling: handles ray tracing, projecting rays from the camera into the 3D scene
        ray_sampler = RaySamplerSingleImage(data, device='cuda', load_gt_depth=load_gt_depth)
        ray_batch = ray_sampler.get_all() # Contains all rays sampled from the current view
        
        # NeRFool if no global source ray batch exists. 
        if src_ray_batch_glb is None:  ## if src_ray_batch_glb is not set (i.e., view_specific), then src_ray_batch is the current ray_batch
            src_ray_batch = ray_batch
        
        # NerFool+ case, ignore
        else:
            src_ray_batch = src_ray_batch_glb
            
        
        # NeRFool case! 
        if not args.no_attack and args.view_specific and (not args.use_trans_attack or i == 0):  # optimize view-specific adv peturb
            epsilon = torch.tensor(args.epsilon / 255.).cuda() # Normalize epsilon
            alpha = torch.tensor(args.adv_lr / 255.).cuda() # The learning rate for the perturbation
            upper_limit = 1 # Pixel range
            lower_limit = 0 # Pixel range
            
            # If camera perturbation is enabled
            if args.perturb_camera:
                # Initialize the camera's rotation parameters for each source view
                # Either set to 0 or initialized randomly                
                rot_param = torch.zeros(args.num_source_views, 3)  ## 3 rotation degrees
                if not args.zero_camera_init:
                    rot_param.uniform_(-args.rot_epsilon/180 * np.pi, args.rot_epsilon/180 * np.pi)
                rot_param.requires_grad = True
                
                # Initializes the translation parameters for each view
                trans_param = torch.zeros(args.num_source_views, 3)  ## 3 translation distances
                if not args.zero_camera_init:
                    trans_param.uniform_(-args.trans_epsilon, args.trans_epsilon)
                trans_param.requires_grad = True

                # Stores the original camera positions
                src_cameras_orig = src_ray_batch['src_cameras'].clone()  # [1, num_source_views, 34]

            # Initialize perturbation delta uniformly
            delta = init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit)
            
            # Adam used here, ignore
            if args.use_adam:
                if args.perturb_camera:
                    params = [delta, rot_param, trans_param]
                else:
                    params = [delta]
                    
                opt = torch.optim.Adam(params, lr=args.adam_lr)
                scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
                
                if args.use_pcgrad:
                    opt = PCGrad(opt, num_source_views=args.num_source_views)
            
            """----------------PERTURBATION BEGINS------------------"""
            print("Start adversarial perturbation...")

            # Optimization loop
            # Note that this is going to loop for both Adam and non-adam
            for num_iter in range(args.adv_iters): 

                # Perturb camera positions if necessary
                if args.perturb_camera:
                    # Generates new camera positions based on the rotation and translation
                    rot_trans = transform_src_cameras(src_cameras_orig, rot_param, trans_param, args.num_source_views)  # [num_source_views, 3, 4]
                    rot_trans = rot_trans.reshape(-1, 12)  # [num_source_views, 12] 
                    src_ray_batch['src_cameras'] = torch.cat([src_cameras_orig[:,:,:-16], rot_trans.unsqueeze(0), src_cameras_orig[:,:,-4:]], dim=2)

                # Adam used here, Ignore        
                if args.use_adam:
                    loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True)
                    
                    opt.zero_grad()
                    
                    if args.use_pcgrad:
                        opt.pc_backward(loss_dict, major_loss=args.major_loss)
                    else:
                        loss.backward()
                        
                    delta.grad.data *= -1

                    if args.perturb_camera:
                        rot_param.grad.data *= -1
                        trans_param.grad.data *= -1
                        
                    opt.step()
                    scheduler.step()
                
                else:
                    # Optimize the adversarial perturbation
                    # loss_dict contains the individual losses used for perturbation optimization
                    loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True)


                    # Backpropagate the loss to compute gradients with respect to the perturbation delta
                    # Prepares the perturbatino for an update using gradient ascent
                    loss.backward()

                    # Extract gradient from delta
                    grad = delta.grad.detach()

                    # Update delta
                    if args.use_l2:
                        # Normalize the gradients to have unit L2 norm.
                        grad_norm = torch.norm(grad.view(len(delta), -1), p=2, dim=1).clamp(min=1e-20)
                        grad = grad / grad_norm.view(len(delta), 1, 1, 1)

                        perturbed_delta_data = delta.data.detach() + alpha * torch.sign(grad)

                        # Project perturbation delta to have L2 norm less than or equal to eps.
                        change_in_delta = perturbed_delta_data - delta.data.detach()
                        l2_delta_change = change_in_delta.renorm(p=2, dim=0, maxnorm=epsilon)
                        with torch.no_grad(): delta += l2_delta_change
                    else:
                        delta.data = delta.data + alpha * torch.sign(grad)
                    
                    # Reset gradients: Zero out the gradients to avoid accumulation during subsequent updates
                    delta.grad.zero_()
                    
                    # Update rotation and translation using gradient ascent
                    # learning rate is same adv_lr
                    if args.perturb_camera:
                        grad = rot_param.grad.detach()
                        print("Use L2? ", args.use_l2)

                        if args.use_l2:
                            # Normalize the gradients to have unit L2 norm.
                            grad_norm = torch.norm(grad.view(len(rot_param), -1), p=2, dim=1).clamp(min=1e-20)
                            grad = grad / grad_norm.view(len(rot_param), 1, 1, 1)

                            perturbed_rot_param_data = rot_param.data.detach() + args.adv_lr * torch.sign(grad)
                        
                            # Project perturbation delta to have L2 norm less than or equal to eps.
                            change_in_delta = perturbed_rot_param_data - rot_param.data.detach()
                            l2_delta_change = change_in_delta.renorm(p=2, dim=0, maxnorm=args.rot_epsilon / 180 * np.pi)
                            with torch.no_grad(): rot_param.data = rot_param.data + l2_delta_change
                        else: 
                            rot_param.data = rot_param.data + args.adv_lr * torch.sign(grad)

                        rot_param.grad.zero_()

                        grad = trans_param.grad.detach()

                        if args.use_l2:
                            # Normalize the gradients to have unit L2 norm.
                            grad_norm = torch.norm(grad.view(len(trans_param), -1), p=2, dim=1).clamp(min=1e-20)
                            grad = grad / grad_norm.view(len(trans_param), 1, 1, 1)

                            perturbed_trans_param_data = trans_param.data.detach() + args.adv_lr * torch.sign(grad)

                            # Project perturbation delta to have L2 norm less than or equal to eps.
                            change_in_delta = perturbed_trans_param_data - trans_param.data.detach()
                            l2_delta_change = change_in_delta.renorm(p=2, dim=0, maxnorm=args.trans_epsilon / 180 * np.pi)
                            with torch.no_grad(): trans_param.data = trans_param.data + l2_delta_change
                        else:
                            trans_param.data = trans_param.data + args.adv_lr * torch.sign(grad)

                        trans_param.grad.zero_()           
                

                # Clamp Perturbation both epsilon and pixel range
                delta.data = clamp(delta.data, -epsilon, epsilon)
                delta.data = clamp(delta.data, lower_limit - src_ray_batch['src_rgbs'], upper_limit - src_ray_batch['src_rgbs'])  
                
                # Clamp camera parameters: Rotation and Translation
                if args.perturb_camera:
                    rot_param.data = clamp(rot_param.data, torch.tensor(-args.rot_epsilon/180 * np.pi).cuda(), torch.tensor(args.rot_epsilon/180 * np.pi).cuda())
                    trans_param.data = clamp(trans_param.data, torch.tensor(-args.trans_epsilon).cuda(), torch.tensor(args.trans_epsilon).cuda())  
                
        """-------- Export Adversarial Source Images --------"""
        if args.export_adv_source_img:
            adv_src_rgbs = src_ray_batch['src_rgbs'] + delta   # [1, N_views, H, W, 3]
            adv_src_rgbs = adv_src_rgbs[0]  # [N_views, H, W, 3]
            src_rgbs = src_ray_batch['src_rgbs'][0]  # [N_views, H, W, 3]

            """-------- Save Adversarial and Original Images --------"""
            for j in range(adv_src_rgbs.shape[0]):
                adv_src_img = adv_src_rgbs[j,:,:,:]
                adv_src_img = (255 * np.clip(adv_src_img.data.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, 'adv_src_{}_{}.png'.format(i,j)), adv_src_img)
                
                src_img = src_rgbs[j,:,:,:]
                src_img = (255 * np.clip(src_img.data.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, 'src_{}_{}.png'.format(i,j)), src_img)
            
            # sys.exit()

        render_and_evaluate(args, i, model, data, delta, src_ray_batch, ray_sampler, ray_batch, file_id, results_dict, current_scene_results_dict, metrics)


"""OPTIMIZATION BEGINS, SEPERATE INTO ATTACK & EVAL & SAVERESULT"""

if __name__ == '__main__':
    
    # Parsing command-line arguments
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False
    
    # Deterministic sampling - the random ray sampling will be consistent across runs
    args.det = True  ## always use deterministic sampling for coarse and fine samples 

    """------Model Initialization------"""
    # Create IBRNet model, initialize with args
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset

    # Set up output directory
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    # Initialize Project object to handle ray tracing and projecting 3D points into 2D images
    projector = Projector(device='cuda')

    # Ensure only a single scene is being evaluated at a time. 
    # If more than one scene is passed, the script will throw an error
    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]

    # Save result for specific scene
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
    os.makedirs(out_scene_dir, exist_ok=True)

    # Initialize training dataset
    train_dataset = None
    train_loader = None
    
    # This is for NeRFool+ Ignore...
    if not args.no_attack and not args.view_specific:  # optimize generalizable adv perturb across different views
        train_dataset = dataset_dict[args.eval_dataset](args, 'train', scenes=args.eval_scenes)
        train_loader = DataLoader(train_dataset, batch_size=1, worker_init_fn=lambda _: np.random.seed(), num_workers=args.workers, 
                                    pin_memory=True, shuffle=True)

    """------Test Dataset Loading------"""
    # Load based on the dataset and scenes specified
    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    save_prefix = scene_name # For result saving later

    # Creates a dataloader for the test dataset
    # Batch size is 1 because we are processing each test scene one by one
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Stores the total number of batches in the test loader
    total_num = len(test_loader)
    
    """------Results Dictionary and Metrics Initialization"""
    results_dict = {scene_name: {}}

    current_scene_result_dict = {
        'sum_coarse_psnr': 0, # Low-resolution
        'sum_fine_psnr': 0,   #High-resolution
        'running_mean_coarse_psnr': 0,
        'running_mean_fine_psnr': 0,
        'sum_coarse_lpips': 0,
        'sum_fine_lpips': 0,
        'running_mean_coarse_lpips': 0,
        'running_mean_fine_lpips': 0,
        'sum_coarse_ssim': 0,
        'sum_fine_ssim': 0,
        'running_mean_coarse_ssim': 0,
        'running_mean_fine_ssim': 0
    }

    """------TensorFlow Campatibility Setup-----"""
    # For LPIPS, PSNR, and SSIM Calculations, Ignore
    if "2." in tf.__version__[:2]:
        tf.compat.v1.disable_eager_execution()
        tf = tf.compat.v1
    
    # TensorFlow placeholders and metrics initialization
    # Initialize PSNR, LPIPS, SSIM
    # Ignore
    metrics = {
        'pred_ph': tf.placeholder(tf.float32),
        'gt_ph': tf.placeholder(tf.float32)
    }
    metrics['distance_t'] = lpips_tf.lpips(metrics['pred_ph'], metrics['gt_ph'], model='net-lin', net='vgg'),
    metrics['ssim_tf'] = tf.image.ssim(metrics['pred_ph'], metrics['gt_ph'], max_val=1.)
    metrics['psnr_tf'] = tf.image.psnr(metrics['pred_ph'], metrics['gt_ph'], max_val=1.)
    
    # Initialize Criterion
    # Used to calculate the loss during optimization
    criterion = Criterion()
    
    #Check whether GT depth is avaliable 
    if args.gt_depth_path:
        # Depth maps will be used during evaluation/perturbation optimization
        load_gt_depth = True
    else:
        load_gt_depth = False
    
    """------ Initialize delta for NerFool+"""
    src_ray_batch = None
    src_ray_batch = initialize_delta(args, train_loader, load_gt_depth)

    """------ Run Main NerFool Pipeline"""
    Run_NerFool(args, model, test_loader, src_ray_batch, out_scene_dir, load_gt_depth, results_dict, current_scene_result_dict, metrics)