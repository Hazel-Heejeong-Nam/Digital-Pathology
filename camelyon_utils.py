############## tools for CAMELYON dataset #################

import pandas as pd
import numpy as np
import openslide
import cv2
import os
import math
import multiprocessing as mp
import argparse
import matplotlib.pyplot as plt
from itertools import repeat

def check_camelyon(args):
    base_root = '/nfs/thena/shared/camelyon/CAMELYON{}/{}'.format(args.dataset,args.mode)
    try : 
        os.mkdir('c{}_image'.format(args.dataset))
    except :
        pass
    tag = {'images' : '', 'background_tissue' : '_tissue', 'masks' : '_mask'}
    for i in range(40,41):
        img = openslide.OpenSlide(os.path.join(base_root, 'normal_0{}{}.tif'.format(i+1, tag[args.mode])))
        lowimg = np.array(img.read_region((0,0), img.level_count -1, img.level_dimensions[-1]))
        if args.mode == 'images':
            pass
        else :
            lowimg = lowimg[:,:,0] * 255
        cv2.imwrite('c{}_image/norm_0{}{}.png'.format(args.dataset,i+1,tag[args.mode]),lowimg)
    
    
    
def patch_generation(args): 
    args.wsi_root = '/nfs/thena/shared/camelyon/CAMELYON{}/images'.format(args.dataset)
    
    data_list = os.listdir(args.wsi_root)
    if args.multiprocess :
        p = mp.Pool(20)
        p.starmap(extract_patch,zip(data_list,repeat(args)))
    else : 
        for item in data_list :
            extract_patch(item, args)
            
            
            
def extract_patch(item, args):
    args.imgName = item.split('.')[0]
    existing_list = os.listdir(os.path.join(args.save_dir, 'CAMELYON{}'.format(args.dataset)))
    for items in existing_list:
        isexist = items.find(args.imgName)
        if isexist != -1:
            print('pass')
            return
    print(item)
    wsi, mask = get_mask(args, item)
    patchbag = mapping(args,wsi, mask)
    if args.saveimg : 
        view_patch(args, patchbag)


    
def get_mask(args,item):
    wsi = openslide.OpenSlide(os.path.join(args.wsi_root,item))
    if args.masklevel == None :
        args.masklevel = wsi.level_count -1

    w,h = wsi.level_dimensions[args.masklevel] 
    maskimg = np.array(wsi.read_region((0,0),args.masklevel, (w,h) )) 
    cvimg = cv2.cvtColor(maskimg[:,:,:3],cv2.COLOR_RGB2YCrCb)
    _, th1 = cv2.threshold(cvimg[:, :, 1], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(cvimg[:, :, 2], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = th1 | th2                     
    return wsi, mask



def mapping(args, wsi, mask):
    hp = int(args.patch_size * args.mpp / float(wsi.properties['openslide.mpp-x'])) # patch size for high resolution
    mp = int(hp / wsi.level_downsamples[args.masklevel]) # patch size for low resolution otsu mask
    w,h = wsi.level_dimensions[args.masklevel] 
    
    patchbag = []
    exception = 0
    for i in range(math.floor(w/mp)):
        for j in range(math.floor(h/mp)):
            try : 
                if mask[j*mp:(j+1)*mp, i*mp:(i+1)*mp].sum() >= mp*mp* args.mask_threshold*255 :
                    patch = wsi.read_region((hp*i, hp*j),0,(hp,hp)).resize((args.patch_size, args.patch_size))
                    patchbag.append(np.array(patch)[:,:,:3])  
            except :
                exception += 1 
                
    if patchbag == [] :
        print(f'Cannot extract patch from {args.imgName}')
    elif exception != 0 :
        print(f'{exception} patches are unable to read from {args.imgName}')
    else :
        if args.savepatchbag:
            np.save(os.path.join(args.save_dir, 'CAMELYON{}/{}_patch_{}'.format(args.dataset, args.imgName, len(patchbag))), np.array(patchbag))
        print(f'Patch collection is done for {args.imgName}')
    return np.array(patchbag)



def view_patch(args, patchbag):
    patchnum = patchbag.shape[0]
    if patchnum <= 800:
        plt.figure(figsize = (60,20))
        for i in range(patchnum):
            plt.subplot(20,40, i+1)
            plt.imshow(patchbag[i])
            plt.axis('off')
        plt.savefig('result/{}_PATCH_{}.png'.format(args.imgName, patchnum))
    else : 
        print('Notification : Too many patches to visualize. Save npy file instead.')
        #np.save('result/{}_PATCH_{}'.format(args.imgName, patchnum), patchbag)
  
  
  
def c_datalist(args): 
    base = os.path.join(args.save_dir, 'CAMELYON{}'.format(args.dataset))
    list = os.listdir(base)
    ref = pd.read_csv('reference.csv', names=['name', 'class', 'subclass', 'binary'])
    train_list = []
    test_list = []
    
    for item in list :
        if 'test' in item :
            _class = ref['class'][ref['name'].str.contains(item.split('_patch')[0])]
            dict = {
                    "image": os.path.join(base, item) , 
                    "label": 0 if _class.item() == 'normal' else 1               
                }
            test_list.append(dict)
        else :
            dict = {
                    "image": os.path.join(base, item) , 
                    "label": 0 if 'normal' in item else 1                 
                }
            train_list.append(dict)       
    return train_list, test_list
  
  
  
def parse_args():
    parser = argparse.ArgumentParser(description="Image Preprocessing for Digital Pathology")

    # data
    parser.add_argument("--dataset", default = '16')
    parser.add_argument("--wsi_root", default = None)
    parser.add_argument("--save_dir", default="/nfs/thena/shared/camelyon_patch", help="path to root folder of images")

    # detail
    parser.add_argument("--patch_size", default = 256, type =int, help="The number of pixel that you want for the patches")
    parser.add_argument("--masklevel", default = 4, type= int)
    parser.add_argument("--mask_threshold", default = 0.75, type= float, help = 'minimum portion of foreground to keep certain patch')
    parser.add_argument("--mpp", default= 1, type = int)

    # mode
    parser.add_argument("--saveimg", default=False, help="Save images during process")
    parser.add_argument("--savepatchbag", default = False, help = "Save patchbag.npy to directory")
    parser.add_argument("--multiprocess", default= False)
    parser.add_argument("--mode", default = 'images', type=str)

    args = parser.parse_args()
    return args


if __name__ =="__main__":
    args = parse_args()
    #patch_generation(args)
    train, test = c_datalist(args)
    print(train)
    print(test)
