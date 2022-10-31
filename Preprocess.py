import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse


def get_rough_contour(imgPath , args):

    img = openslide.OpenSlide(imgPath)
    dims = img.level_dimensions[-1]
    img = img.read_region((0,0),img.level_count-1, dims)
    cvimg = cv2.cvtColor(np.array(img),cv2.COLOR_RGBA2GRAY)
    #thr = cv2.adaptiveThreshold(cvimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,-1)
    thr = cv2.adaptiveThreshold(cvimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,99,-5)
    thr = np.full(thr.shape, 255, dtype = np.uint8) - thr
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contimg = cv2.drawContours(cvimg, contours, -1, (0,255,0), 3)
    if args.saveimg:        
        cv2.imwrite('contour.png', contimg)
    
    return thr, contours, cvimg, contimg

def get_ROI(contours, contimg, args):
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(contimg,[box],0,(0,0,255),2)
        if args.saveimg:        
            cv2.imwrite('box.png', contimg)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Image Preprocessing for Digital Pathology")
    parser.add_argument("--saveimg", default=False, help="save images during process")
    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    
    args = parse_args()
    args.saveimg = True
    imgPath = '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/CNB_2018-2019(4th)/CODIPAI-THCB-SS-00822/CODIPAI-THCB-SS-00822-S-CN-01.svs'
    
    hr, contours, cvimg, contimg = get_rough_contour(imgPath, args)
    get_ROI(contours, contimg, args)
