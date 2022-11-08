import os
import torch
import random
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90
)

# creat data dictionary with npy image and label
def datalist(foldername):
    dataset = []
    for root, dir, files in os.walk(os.path.join('/nfs/thena/shared/thyroid_patch',foldername)):
        for file in files:
            dict = {
                    "image": os.path.join(root,file), 
                    "label": float(file.split('_')[-1].split('.')[0])                    
                }
            dataset.append(dict)

    return dataset

# Get random patches from data
def RandPatch(train_patch, data):
    random.seed(0)
    data = data.permute((0,1,4,2,3))   # reshape (1,N,W,H,C) -> (1,N,C,W,H)
    cnt = data.shape[1]

    idx = random.sample(range(cnt),train_patch)
    data = data[:,idx,:,:,:]  # (1,50,C,W,H)

    return data
