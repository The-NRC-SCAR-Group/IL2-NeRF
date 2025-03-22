#!/usr/bin/env bash

# deepvoxels
# download the deepvoxels data [synthetic_scenes.zip]
# https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH
gdown https://drive.google.com/uc?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl
unzip synthetic_scenes.zip -d deepvoxels

# nerf realistic 360
# download the nerf synthetic data [nerf_synthetic.zip]
# https://drive.google.com/drive/folders/1PXEDEECbxhEFG_PkMekJuqVRZB4roDR_
gdown https://drive.google.com/uc?id=1zgf9k57nKBXM6gjIjruILrJjPBn7VbTd
unzip nerf_synthetic.zip

# real forward facing
# download NeRF LLFF test data [nerf_llff_data.zip]
# https://drive.google.com/drive/folders/1PXEDEECbxhEFG_PkMekJuqVRZB4roDR_
gdown https://drive.google.com/uc?id=1wyp-pmhftUgvFkAdCRbXAW7i2XKc4ikH
unzip nerf_llff_data.zip



