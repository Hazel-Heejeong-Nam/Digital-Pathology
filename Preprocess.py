import openslide
import numpy as np
import cv2
import argparse
import math


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
    
    ext = imgPath.split('.')[-1]
    img = openslide.OpenSlide(imgPath)
    args.mpp = float(img.properties['openslide.mpp-x'])
    if ext =='tiff':
        dims = img.level_dimensions[-2]
        lowimg = np.array(img.read_region((0,0),img.level_count-2, dims))    
        args.ratio = img.level_downsamples[-2]
    else :
        dims = img.level_dimensions[-1]
        lowimg = np.array(img.read_region((0,0),img.level_count-1, dims))    
        args.ratio = img.level_downsamples[-1]

    cvimg = cv2.cvtColor(lowimg,cv2.COLOR_RGBA2GRAY)
    thr = cv2.adaptiveThreshold(cvimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,99,-5)
    thr = np.full(thr.shape, 255, dtype = np.uint8) - thr
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if args.saveimg:  
        contimg = cv2.drawContours(cvimg, contours, -1, (0,255,0), 3)      
        cv2.imwrite('contour.png', contimg)
    
    return contours, lowimg, img

def get_ROI(args, contours, lowimg):
    
    """
    Description : 
        Find a 'max contour' which has the largest contour area.
        Draw a bounding box of 'max contour.'
        Crop the box and make 'otsu threshold mask' with that cropped image.
        
    Args :
        args
        contours (tuple) : Contours from 'get_rough_contour'
        lowimg (numpy array) : low resolution WSI image in RGBA color space.
        
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
    crop_img = lowimg[y: y + h, x: x + w]

    # CREATE OTSU MASK
    ycrcb = cv2.cvtColor(crop_img,cv2.COLOR_RGB2YCrCb)
    _, th1 = cv2.threshold(ycrcb[:, :, 1], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(ycrcb[:, :, 1], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = th1 | th2
    
    if args.saveimg:       
        cv2.rectangle(lowimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite('box.png', lowimg)
        cv2.imwrite('crop_lowres.png', crop_img)
        cv2.imwrite('mask_lowres.png', mask)

    return x,y,w,h , mask

def foreground_patches(args, x,y,w,h,mask, img):
    
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
    
    hp = int(args.patch_size * args.desired_mpp / args.mpp) # patch size for high resolution
    mp = int(hp / args.ratio) # patch size for low resolution otsu mask
    x_start = int(x* args.ratio)
    y_start = int(y* args.ratio)
    patchbag = []
    
    for i in range(math.floor(w/mp)):
        for j in range(math.floor(h/mp)):
            sum = mask[j*mp:(j+1)*mp, i*mp:(i+1)*mp].sum()
            if sum >= mp*mp*args.mask_threshold*255:
                loc_x = int(x_start + i*hp) # left-top x location in level0 image
                loc_y = int(y_start + j*hp) # left-top y location in level0 image
                patch = img.read_region((loc_x,loc_y),0,(hp,hp))
                patchbag.append(np.array(patch)[:,:,:3])
    
    print(len(patchbag))

    return np.array(patchbag)
                    
    
def parse_args():
    
    parser = argparse.ArgumentParser(description="Image Preprocessing for Digital Pathology")
    parser.add_argument("--saveimg", default=False, help="save images during process")
    parser.add_argument("--desired_mpp", default=1.0, type = float)
    parser.add_argument("--mpp", default = None, type = float)
    parser.add_argument("--patch_size", default = 256, type =int)
    parser.add_argument("--ratio", default = None, type= float)
    parser.add_argument("--mask_threshold", default = 0.15, type= float)

    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    
    args = parse_args()
    args.saveimg = True
    imgPath = '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/AITRICS_Core_Needle_Biopsy_annotated_SS_THCB_1959-2407(447)/CODIPAI-THCB-SS-02364-S-CN-01.ndpi'
    
    contours, lowimg , img= get_rough_contour(args, imgPath)
    x,y,w,h , mask = get_ROI(args, contours, lowimg)
    foreground_patches(args, x,y,w,h,mask, img)
