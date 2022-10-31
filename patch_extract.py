import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist
from monai.data.wsi_reader import WSIReader
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    MapTransform,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
    ScaleIntensityRanged,
    SplitDimd,
    ToNumpyd,
)


def save_patch(args, destination, key):
    
    dataset = load_decathlon_datalist(
            data_list_file_path= args.dataset_json,
            data_list_key=key,
            base_dir= args.data_root,
        )

    transform = Compose(
        [
            LoadImaged(keys=["image"], reader=WSIReader(backend='cucim', level = 1) ,dtype=np.uint8, image_only=True),
            
            GridPatchd(
                keys=["image"],
                patch_size=(args.tile_size, args.tile_size),
                #num_patches=args.num_patch,
                threshold=0.8 * 3 * 255 * args.tile_size * args.tile_size,
                sort_fn="min",
                pad_mode=None,
                constant_values=255,
            ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            #ScaleIntensityRanged(keys=["image"], a_min=np.float32(255), a_max=np.float32(0)),
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5),
            ToNumpyd(keys=["image"], dtype = np.float16 ),
        ]
    ) 

    data_tr =  Dataset(data=dataset, transform=transform)

    for i in range(len(data_tr)):
        start_time = time.time()
        wsi =  data_tr.__getitem__(i)
        for j in range(len(wsi)):
            slide = wsi[j]["image"].reshape((1, 3, args.tile_size, args.tile_size))
            if j==0 :
                npimg = slide
            else :
                npimg = np.concatenate((npimg,slide), axis=0)
        imgName = wsi[0]["path"].split('/')[-1].split('.')[0] + '_fullpatch2'
        np.save(os.path.join(destination,imgName), npimg)
        print('{} saved , {} sec'.format(imgName,time.time() - start_time))


def parse_args():

    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) example of classification from WSI.")
    parser.add_argument("--data_root", default="/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK", help="path to root folder of images")
    parser.add_argument("--dataset_json", default='test.json', type=str, help="path to dataset json file")
    parser.add_argument("--num_patch", default=44, type=int, help="number of patches (instances) to extract from WSI image")
    parser.add_argument("--tile_size", default=256, type=int, help="size of square patch (instance) in pixels")
    parser.add_argument("--test", default = False, type= bool )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train_dir = '/home/hazel/storage/train'
    val_dir = '/home/hazel/storage/val'
    test_dir = '/home/hazel/storage/test/image'


    ##test
    args.test = True

    if args.test :
        save_patch(args, test_dir, "test" )
    else :
        save_patch(args, train_dir, "training")
        print('All train data save in {}'.format(train_dir))
        save_patch(args, train_dir, "validation")
        print('All train data save in {}'.format(train_dir))
