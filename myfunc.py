import os
import torch
import random
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90
)
from preprocessing_mp import * 
import inspect
import os
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.data import PatchWSIDataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import iter_patch_position
from monai.data.wsi_reader import BaseWSIReader, WSIReader
from monai.transforms import ForegroundMask, Randomizable, apply_transform
from monai.utils import convert_to_dst_type, ensure_tuple_rep
from monai.utils.enums import CommonKeys, ProbMapKeys, WSIPatchKeys
from itertools import starmap

# creat data dictionary with npy image and label
def datalist(foldername,oversample = 3):
    dataset = []
    for root, dir, files in os.walk(os.path.join('/nfs/thena/shared/thyroid_patch',foldername)):
        for file in files:
            #####
            label = float(file.split('_')[-1].split('.')[0])         
            dict = {
                    "image": os.path.join(root,file), 
                    "label": label                 
                }
            if label == 1 and foldername =="train":
                for i in range(oversample):
                    dataset.append(dict)
            else :
                dataset.append(dict)

    return dataset

def WSIdatalist(foldername):
    dataset = []
    for root, dir, files in os.walk(os.path.join('/nfs/thena/shared/thyroid_patch',foldername)):
        for file in files:
            #####
            label = float(file.split('_')[-1].split('.')[0])         
            dict = {
                    "image": os.path.join(root,file), 
                    "label": label,             
                }
            dataset.append(dict)
    return dataset



# 근데 이거 batchsize 1일때만 돌아갈듯 생각해보니까 collate function도 없애버렸는데
# Get random patches from data
def RandPatch(train_patch, data):
    data = data.permute((0,1,4,2,3))   # reshape (1,N,W,H,C) -> (1,N,C,W,H)
    augmented = []
    aug = Compose(
    [
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandRotate90(prob=0.5, max_k=3)
    ]
    )
    cnt = data.shape[1]
    for i in range(train_patch):    
        idx = random.randint(0,cnt-1)
        newdata = aug(data[:,idx,::].squeeze()).unsqueeze(0)
        augmented.append(newdata)
    newdata = torch.cat(augmented,0).unsqueeze(0)

    return newdata

class SlidingMaskPatchWSIDataset(Randomizable, PatchWSIDataset):

    def __init__(
        self,
        data: Sequence,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_level: Optional[int] = None,
        mask_level: int = 0,
        transform: Optional[Callable] = None,
        include_label: bool = True,
        additional_meta_keys: Sequence[str] = (ProbMapKeys.LOCATION, ProbMapKeys.SIZE, ProbMapKeys.COUNT),
        reader="cuCIM",
        **kwargs,
    ):
        super().__init__(
            data=[],
            patch_size=patch_size,
            patch_level=patch_level,
            transform=transform,
            include_label=include_label,
            additional_meta_keys=additional_meta_keys,
            reader=reader,
            **kwargs,
        )
        self.mask_level = mask_level
        # Create single sample for each patch (in a sliding window manner)
        self.data: list
        self.image_data = list(data)
        print('Preprocessing start')
        for sample in self.image_data:
            print('Patch generating...')
            patch_samples = self._evaluate_patch_locations(sample)
            self.data.extend(patch_samples)

    def _get_box(self, mask, kernelsize):
        dilation_image = cv2.dilate(mask, np.ones((kernelsize, kernelsize), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxcnt = 0
        maxidx = 0

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > maxcnt : 
                maxcnt = area
                maxidx = i
        return cv2.boundingRect(contours[maxidx])

    def _evaluate_patch_locations(self, sample):
        """Calculate the location for each patch in a sliding-window manner"""
        patch_size = self._get_size(sample)
        patch_level = self._get_level(sample)
        wsi_obj = self._get_wsi_object(sample)
        mask_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, self.mask_level)
        patch_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, patch_level)
        # load the entire image at level=mask_level
        wsi, _ = self.wsi_reader.get_data(wsi_obj, level=self.mask_level)
        stride = int(patch_size[0] * patch_ratio / mask_ratio)
        kernelsize = int(120/np.log2(mask_ratio / patch_ratio))*2 

        # create the foreground tissue mask and get all indices for non-zero pixels
        mask = np.squeeze(convert_to_dst_type(ForegroundMask(hsv_threshold={"S": "otsu"})(wsi), dst=wsi)[0])
        y, x, h, w = self._get_box(mask, kernelsize) 

        # ROI coordinate으로 stride 넣어서 crop한 뒤 nonzero인 value들은 1로 만들어줘야함
        mask = mask[x:x+w+1:stride, y:y+h+1:stride]

        patch_locations = np.array(np.vstack(mask.nonzero()).T * stride + np.array([x,y])) * mask_ratio / patch_ratio # patch level에서의 left top 좌표


        ###########


        # fill out samples with location and metadata
        sample[WSIPatchKeys.SIZE.value] = patch_size
        sample[WSIPatchKeys.LEVEL.value] = patch_level
        sample[ProbMapKeys.NAME.value] = os.path.basename(sample[CommonKeys.IMAGE])
        sample[ProbMapKeys.COUNT.value] = len(patch_locations)
        sample[ProbMapKeys.SIZE.value] = np.array(self.wsi_reader.get_size(wsi_obj, self.mask_level))
        return [
            {**sample, WSIPatchKeys.LOCATION.value: np.array(loc)}
            for loc in zip(patch_locations)
        ]

