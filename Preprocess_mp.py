import openslide
import numpy as np
import cv2
import argparse
import math
import matplotlib.pyplot as plt
from monai.data import load_decathlon_datalist
import os
import time
import multiprocessing as mp

def get_rough_contour(args, imgPath):
    
    """
    Description : 
        Load a single WSI at the highest level (the lowest resolution).
        Get outermost contours with padding so that slightly scattered tissue can be considered as one.
        
    Args :
        args
        imgPath (str) : Image path from json file
        
    Returns :
        contours (tuple) : A tuple containing every outermost contour
        lowimg (numpy array) : low resolution WSI image in RGBA color space 
        img (Openslide) 
        
    """
    
    ext = imgPath.split('.')[-1] # expected : 'ndpi','svs','tiff'
    img = openslide.OpenSlide(imgPath)
    args.mpp = float(img.properties['openslide.mpp-x']) # micron-per-pixel in level 0

    dims = img.level_dimensions[-2]
    lowimg = np.array(img.read_region((0,0),img.level_count-2, dims))    
    args.ratio = img.level_downsamples[-2]
    #kernelsize = int(120/np.log2(args.ratio))+10
    kernelsize = int(120/np.log2(args.ratio))*2

    cvimg = cv2.cvtColor(lowimg[:,:,:3],cv2.COLOR_RGB2YCrCb)
    _, th1 = cv2.threshold(cvimg[:, :, 1], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(cvimg[:, :, 2], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = th1 | th2
    dilation_image = cv2.dilate(mask, np.ones((kernelsize, kernelsize), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if args.saveimg:  
        contimg = cv2.drawContours(cvimg, contours, -1, (0,255,0), 3)      
        cv2.imwrite('result/{}/{}_CONTOUR.png'.format(args.imgName,args.imgName), contimg)
        cv2.imwrite('result/{}/{}_MASK.png'.format(args.imgName,args.imgName), mask)

    return contours, mask, img, dims , lowimg

def get_ROI(args, contours, mask, lowimg):
    
    """
    Description : 
        Find a 'max contour' which has the largest contour area.
        Draw a bounding box of 'max contour.'
        Crop the box and make 'otsu threshold mask' with that cropped image.
        
    Args :
        args
        contours (tuple) : Contours from 'get_rough_contour'
        lowimg (numpy array) : low resolution WSI image in RGBA color space
        
    Returns :
        x,y,w,h (int) : left-top (x,y) coordinate, width, and height of the bounding box
        mask (numpy array) : low-resolution otsu-threshold mask based on lowimg
        
    """
    
    maxcnt = 0
    maxidx = 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > maxcnt : 
            maxcnt = area
            maxidx = i
    x,y,w,h = cv2.boundingRect(contours[maxidx])
    crop_img = mask[y: y + h, x: x + w]
    
    if args.saveimg:       
        cv2.rectangle(lowimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite('result/{}/{}_BOX.png'.format(args.imgName,args.imgName), lowimg)
        cv2.imwrite('result/{}/{}_CROP.png'.format(args.imgName,args.imgName), crop_img)

    return x,y,w,h , crop_img


def otsu_foreground_patches(args, x,y,w,h,mask, img):
    
    """
    Description : 
        Map the low resolution mask to the high resolution image.
        Generate foreground patches with given mpp(micron-per-pixel) and patch-size(pixel).
        
    Args :
        args
        x,y,w,h (int) : left-top (x,y) coordinate, width, and height of the bounding box (low-resolution)
        mask (numpy array) : low-resolution otsu-threshold mask (low-resolution)
        img (Openslide) 
        
    Returns :
        patchbag (numpy array) : An array with stacked foreground patches
        
    """
    
    hp = args.patch_size * args.desired_mpp / args.mpp # patch size for high resolution
    mp = int(hp / args.ratio) # patch size for low resolution otsu mask
    x_start = x* args.ratio
    y_start = y* args.ratio
    patchbag = []
    print('Start collecting patch')
    for i in range(math.floor(w/mp)):
        for j in range(math.floor(h/mp)):
            sum = mask[j*mp:(j+1)*mp, i*mp:(i+1)*mp].sum()
            if sum >= mp*mp*args.mask_threshold*255:
                loc_x = int(x_start + i*hp) # left-top x location in level0 image
                loc_y = int(y_start + j*hp) # left-top y location in level0 image
                patch = img.read_region((loc_x,loc_y),0,(int(hp),int(hp))).resize((args.patch_size, args.patch_size))
                patchbag.append(np.array(patch)[:,:,:3])
    
    print('Total {} patches are collected'.format(len(patchbag)))
    return np.array(patchbag)


def hard_threshold_patch(args, x, y, w, h, img):

    """
    Description : 
        Generate foreground patches with given mpp(micron-per-pixel) and patch-size(pixel) using fixed value.
        
    Args :
        args
        x,y,w,h (int) : left-top (x,y) coordinate, width, and height of the bounding box (low-resolution)
        img (Openslide) 
        
    Returns :
        patchbag (numpy array) : An array with stacked foreground patches
        
    """
    x_start = int(x* args.ratio)
    y_start = int(y* args.ratio)
    hp = int(args.patch_size * args.desired_mpp / args.mpp)
    print('Start collecting patch')
    patchbag = []
    for i in range(math.ceil(w*args.ratio/hp)):
        for j in range(math.ceil(h*args.ratio/hp)):
                loc_x = int(x_start + i*hp) # left-top x location in level0 image
                loc_y = int(y_start + j*hp) # left-top y location in level0 image
                patch = img.read_region((loc_x,loc_y),0,(hp,hp)).resize((args.patch_size, args.patch_size))
                patch = np.array(patch)[:,:,:3]
                if patch.sum() < 0.85 * 3 * 255 * args.patch_size * args.patch_size:
                    patchbag.append(patch)

    print('Total {} patches are collected'.format(len(patchbag)))
    return np.array(patchbag)

def full_hardthreshold(args,img):

    """
    Description : 
        Generate foreground patches with given mpp(micron-per-pixel) and patch-size(pixel) using fixed value.
        
    Args :
        args
        x,y,w,h (int) : left-top (x,y) coordinate, width, and height of the bounding box (low-resolution)
        img (Openslide) 
        
    Returns :
        patchbag (numpy array) : An array with stacked foreground patches
        
    """
    dim = img.level_dimensions[0]
    hp = int(args.patch_size * args.desired_mpp / args.mpp)
    print('Start collecting patch')
    patchbag = []
    for i in range(math.floor(dim[0]/hp)-1):
        for j in range(math.floor(dim[1]/hp)-1):
            loc_x = i*hp # left-top x location in level0 image
            loc_y = j*hp # left-top y location in level0 image
            patch = img.read_region((loc_x,loc_y),0,(hp,hp)).resize((args.patch_size, args.patch_size))
            patch = np.array(patch)[:,:,:3]
            if patch[0,0].sum() ==0 or patch[0,255].sum() ==0 or patch[255,0].sum() ==0 or patch[255,255].sum() ==0:
                pass
            else :
                if patch.sum() < 0.85 * 3 * 255 * args.patch_size * args.patch_size:
                    patchbag.append(patch)

    print('Total {} patches are collected'.format(len(patchbag)))
    return np.array(patchbag)


def view_patch(patchbag):
    """
    Description : 
        Visualize extracted patches.
        
    Args :
        patchbag (numpy array): The array with stacked foreground patches

    """
    patchnum = patchbag.shape[0]
    plt.figure(figsize = (60,20))
    for i in range(patchnum):
        plt.subplot(15,35, i+1)
        plt.imshow(patchbag[i])
        plt.axis('off')

    plt.savefig('result/{}/{}_PATCH_{}_new.png'.format(args.imgName,args.imgName,patchnum))

def parse_args():
    
    parser = argparse.ArgumentParser(description="Image Preprocessing for Digital Pathology")
    parser.add_argument("--data_root", default="/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK", help="path to root folder of images")
    parser.add_argument("--saveimg", default=False, help="Save images during process")
    parser.add_argument("--desired_mpp", default=1.0, type = float, help="MPP that you want for the patches")
    parser.add_argument("--mpp", default = None, type = float, help="Current mpp at the level 0")
    parser.add_argument("--patch_size", default = 256, type =int, help="The number of pixel that you want for the patches")
    parser.add_argument("--ratio", default = None, type= float, help ='')
    parser.add_argument("--mask_threshold", default = 0.4, type= float)
    parser.add_argument("--dataset_json", default='large_kernel.json',type= str )
    parser.add_argument("--label", default= None, type = int)
    parser.add_argument("--imgName", default=None , type= str)

    args = parser.parse_args()
    
    return args
    
def process(data):
    start_time = time.time()
    args.imgName = data['image'].split('/')[-1].split('.')[0]

    #pass if preprocessing is already done
    for root, directories, files in os.walk('/nfs/thena/shared/thyroid_patch/train'):
        for file in files :
            isexist = file.find(args.imgName)
            if isexist != -1:
                print('pass')
                return 

    label = str(data['label'])
    print("*************{}**************".format(args.imgName))
    if args.saveimg :
        try :
            os.mkdir('result/{}'.format(args.imgName))
        except:
            pass
    contours, mask , img, dims, lowimg= get_rough_contour(args, data['image'])
    x,y,w,h, cropmask = get_ROI(args, contours, mask, lowimg)

# choose threshold method

    #patchbag = hard_threshold_patch(args, x, y, w, h, img)
    #patchbag = otsu_foreground_patches(args, x,y,w,h,mask, img)
    patchbag = full_hardthreshold(args,img)

    np.save(os.path.join('/nfs/thena/shared/thyroid_patch/train',args.imgName +'_{}_{}'.format(patchbag.shape[0],label)), patchbag)
    if args.saveimg :
        view_patch(patchbag)
    print('{} sec'.format(time.time() - start_time))


def main(args):
    
    train_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="training",
        base_dir=args.data_root,
    )
    validation_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="validation",
        base_dir=args.data_root,
    )
    
    #preprocess training dataset
    p = mp.Pool(20)
    p.map(process, train_list)

"""
train patch 뽑을 때는 p.map(process, train_list), process에서 저장경로 thyroid_patch/train 으로
validation patch 뽑을 때는 p.map(process, validation_list), process에서 저장경로 thyroid_patch/test 로
이미 전처리 끝난 파일 체크하는 os.walk 경로도 train/test에 맞게 바꿔주기
재전처리인 경우 json 제대로 들어갔는지 확인하기

"""


if __name__ == "__main__":
    
    args = parse_args()
    #args.saveimg = True

    main(args)
    

