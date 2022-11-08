import os
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_mp import *
import multiprocessing as mp
import subprocess

def check_dist(folder, min_ ,max_):
    num_bag = []
    underMIN = []
    overMAX = []
    for root, directories, files in os.walk(os.path.join('/nfs/thena/shared/thyroid_patch',folder)):
        for file in files:
            patchnum = int(file.split('.')[-2].split('_')[-2])
            if patchnum < min_ :
                underMIN.append(os.path.join(root,file))
            elif patchnum > max_ :
                overMAX.append(os.path.join(root,file))
            num_bag.append(patchnum)
    
    print('under {} : {}, over {}: {} while total number of file : {}'.format(min_, len(underMIN), max_, len(overMAX), len(num_bag)))
    print('minimum patch num in {} data : {}'.format(folder, min(num_bag)))
    print('maximum patch num in {} data : {}'.format(folder, max(num_bag)))            
    plt.hist(num_bag,bins=30)
    plt.savefig('{}_histogram.png'.format(folder))
    plt.show()
    plt.close()
    return underMIN, overMAX


def view_patch(args, list):
    #for pool.map
    # args = parse_args()
    # args.saveimg = True

    for data in list:
        bag = np.load(data)
        args.imgName = data.split('/')[-1].split('.')[0].split('_')[-3]
        patchnum = bag.shape[0]

        try : 
            os.mkdir('result/{}'.format(args.imgName))
        except:
            pass
        plt.figure(figsize = (40,40))
        for i in range(patchnum):
            plt.subplot(40,40, i+1)
            plt.imshow(bag[i])
            plt.axis('off')
            plt.savefig('result/{}/{}_PATCH_{}_new.png'.format(args.imgName,args.imgName,patchnum))
        plt.close()

        print('finding {}....'.format(args.imgName))
        for root, directories, files in os.walk('/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK'):
            for file in files :
                name, ext = os.path.splitext(file)
                if 'Jung_DB' in root: continue
                if ext not in ['.ndpi','.svs','.tiff']: continue 
                
                isexist = file.find(args.imgName)
                if isexist != -1:
                    print('**********{}**********'.format(args.imgName))

                    #remove file in thyroid_patch for re-preprocessing
                    subprocess.call(['rm',data])

                    contours, mask , img, dims, lowimg= get_rough_contour(args,  os.path.join(root,file))
                    x,y,w,h, cropmask = get_ROI(args, contours, mask, lowimg)
                    break



if __name__ == "__main__":

    args = parse_args()
    args.saveimg = True
    print_ = True

    under1, over1 = check_dist('train',20,400)
    under2, over2 = check_dist('test', 20, 400)
    

    # p = mp.Pool(20)
    # p.map(view_patch, under1)
    view_patch(args, under2)

