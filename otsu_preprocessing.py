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
    
    #ext = imgPath.split('.')[-1] # expected : 'ndpi','svs','tiff'
    img = openslide.OpenSlide(imgPath)

    args.mask_level = img.level_count-2 if args.mask_level == None else args.mask_level
    dims = img.level_dimensions[args.mask_level]   
    args.mask_ratio = img.level_downsamples[args.mask_level]
    args.patch_ratio = img.level_downsamples[args.patch_level]
    kernelsize = int(120/np.log2(args.mask_ratio / args.patch_ratio))*2


    maskimg = np.array(img.read_region((0,0),args.mask_level, dims)) 
    cvimg = cv2.cvtColor(maskimg[:,:,:3],cv2.COLOR_RGB2YCrCb)
    _, th1 = cv2.threshold(cvimg[:, :, 1], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(cvimg[:, :, 2], 16, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = th1 | th2
    dilation_image = cv2.dilate(mask, np.ones((kernelsize, kernelsize), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if args.saveimg:  
        contimg = cv2.drawContours(maskimg, contours, -1, (0,255,0), 3)      
        cv2.imwrite('result/{}/{}_CONTOUR.png'.format(args.imgName,args.imgName), contimg)
        cv2.imwrite('result/{}/{}_MASK.png'.format(args.imgName,args.imgName), mask)

    return contours, mask, img, maskimg

def get_ROI(args, contours, maskimg):

    maxcnt = 0
    maxidx = 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > maxcnt : 
            maxcnt = area
            maxidx = i
    x,y,w,h = cv2.boundingRect(contours[maxidx])
    
    if args.saveimg:       
        cv2.rectangle(maskimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite('result/{}/{}_BOX.png'.format(args.imgName,args.imgName),  maskimg)

    return x,y,w,h


def otsu_foreground_patches(args, x,y,w,h,mask, img):
    patchbag = []
    s = int(args.patch_size * args.patch_ratio / args.mask_ratio) # stride
    cropmask = mask[y:y+h+1:s, x:x+w+1:s]

    if args.selection_method == 'sum':
        for i in range(cropmask.shape[1]):
            for j in range(cropmask.shape[0]):
                sum = mask[j*s + y : (j+1)*s + y, i*s + x : (i+1)*s + x].sum()
                if sum >= s*s * args.mask_threshold*255:
                    loc = (int((i*s + x)* args.mask_ratio), int((j*s + y)* args.mask_ratio))
                    patch = img.read_region(loc,args.patch_level,(args.patch_size, args.patch_size))
                    patchbag.append(np.array(patch)[:,:,:3])

    elif args.selection_method == 'lefttop':

        patch_locations = np.array(np.vstack(cropmask.nonzero()).T * s + np.array([y,x])) * args.mask_ratio 
        for items in patch_locations:
            loc = (int(items[1]), int(items[0])) 
            patch = img.read_region(loc, args.patch_level, (args.patch_size, args.patch_size))
            patchbag.append(np.array(patch)[:,:,:3])

    else :
        print('Error : args.selection should be \'sum\' or \'lefttop\' ')

 
    return np.array(patchbag)

def view_patch(patchbag):

    patchnum = patchbag.shape[0]
    plt.figure(figsize = (40,60))
    for i in range(patchnum):
        plt.subplot(10,10, i+1)
        plt.imshow(patchbag[i])
        plt.axis('off')

    plt.savefig('result/{}/{}_PATCH_{}_new.png'.format(args.imgName,args.imgName,patchnum))
    
def process(data):
    start_time = time.time()
    args.imgName = data['image'].split('/')[-1].split('.')[0]
    args.label = str(data['label'])

    if args.savepatchbag :
        for root, directories, files in os.walk(os.path.join('/nfs/thena/shared/thyroid_patch',args.dataset)):
            for file in files :
                isexist = file.find(args.imgName)
                if isexist != -1:
                    print('pass')
                    return 

    print("*************{}**************".format(args.imgName))
    if args.saveimg :
        try :
            try :
                os.mkdir('result')
                os.mkdir('result/{}'.format(args.imgName)) 
            except :
                os.mkdir('result/{}'.format(args.imgName)) 
        except:
            pass

    
    contours, mask, img, maskimg= get_rough_contour(args, data['image'])
    x,y,w,h= get_ROI(args, contours, maskimg)
    patchbag = otsu_foreground_patches(args, x,y,w,h,mask, img)


    if args.saveimg :
        view_patch(patchbag)
    if args.savepatchbag : 
        np.save(os.path.join('/nfs/thena/shared/thyroid_patch',args.dataset,args.imgName +'_{}_{}'.format(patchbag.shape[0],args.label)), patchbag)
    print('{} sec'.format(time.time() - start_time))


def main(args):
    data_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key=args.dataset,
        base_dir=args.data_root,
    )

    if args.multiprocess :
        p = mp.Pool(20)
        p.map(process, data_list)

    else : 
        for item in data_list :
            process(item)


def parse_args():
    parser = argparse.ArgumentParser(description="Image Preprocessing for Digital Pathology")

    # data
    parser.add_argument("--data_root", default="/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK", help="path to root folder of images")
    parser.add_argument("--dataset_json", default='lymph_newdata.json',type= str )
    parser.add_argument("--label", default= None, type = int)
    parser.add_argument("--imgName", default=None , type= str)

    # detail
    parser.add_argument("--patch_size", default = 256, type =int, help="The number of pixel that you want for the patches")
    parser.add_argument("--mask_ratio", default = None, type= float)
    parser.add_argument("--patch_ratio", default = None, type= float)
    parser.add_argument("--mask_level", default = None, type= int)
    parser.add_argument("--patch_level", default = 1, type= int)
    parser.add_argument("--mask_threshold", default = 0.5, type= float, help = 'minimum portion of foreground to keep certain patch')

    # mode
    parser.add_argument("--saveimg", default=False, help="Save images during process")
    parser.add_argument("--savepatchbag", default = False, help = "Save patchbag.npy to directory")
    parser.add_argument("--multiprocess", default= False)
    parser.add_argument("--dataset", default = 'train', help = "train or test")
    parser.add_argument("--selection_method", default = 'sum', help = 'method for choosing foreground patches')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    args.saveimg = True
    args.mask_level = 3
    args.dataset = 'test'
    #args.selection_method = 'lefttop'
    main(args)
    

