# required package : conda install -c ome bftools
import os
import subprocess
import time
import multiprocessing as mp

def NDPItoTIFF(data):
    file = data[0]
    root = data[1]

    name, ext = os.path.splitext(file)
    if 'Jung_DB' in root : 
        return
    if ext == '.ndpi':
        start_time = time.time()
        outname = name + '_convert.tiff'
        #subprocess.call(['showinf', os.path.join(root, file)])
        subprocess.call(['bfconvert','-series','1','-nobigtiff','-compression','LZW',os.path.join(root, file), os.path.join(root,outname)])
        print('{} saved , {} sec'.format(outname ,time.time() - start_time))

if __name__ == '__main__':
    base = '/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK'
    iter = []
    
    for root, directories, files in os.walk(base):
        for file in files: 
            iter.append([file,root])

    p = mp.Pool(40)
    p.map(NDPItoTIFF,iter)
