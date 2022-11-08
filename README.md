# Digital-Pathology

![image](https://user-images.githubusercontent.com/100391059/200576233-0055a1c9-e329-4103-a73f-4683814a9e10.png)

## main.py
- How to run (possible DDP run command)
``` python
python main.py --distributed --amp --workers 4 --batch_size 2
```
- Description 
> Training code using MIL (Multiple Instance Learning) for WSI (Whole Slide Image) 
> Use resnet34 as a backbone and put output feature into (multi-head) self-attention transformer encoder. (This part is implemented in MONAI MILmodel)
> Based on MONAI tutorial code for digital pathology

## myfunc.py
- Description
  - Necessary file to run main.py
  
- Functions
  - datalist
    > - Create a list of dictionary 
    > - ex) each dictionary looks like :
    > ```
    {
      "image" : '/nfs/thena/shared/thyroid_patch/train/abc.npy'
      "label" : 1.
    }
    ```
    
  - RandPatch
    > - Get fixed amount of patches randomly. 
    > - Default is 20 for training. 
    > - In test phase, we use all patches
    
## Preprocessing_mp.py
- How to run (possible run command)
``` python
python3 preprocessing_mp.py --saveimg
```
- Description
  > 1. Read WSI (expected file extension 'svs', 'tiff', 'ndpi')
  > 2. Extract foregournd patches (default 256 x 256 ) in the region of interest.
  > 3. Save the stack of patches into the destination directory
  
- Functions
  - get_rough_contour
  > Get rough contour with padding (to hold scattered tissue together)
  > Every image is on low-resolution
  > ![image](https://user-images.githubusercontent.com/100391059/200582617-4bbf5946-f5b3-4727-9ac6-764d042fa08d.png)
  
  - get_ROI
  > Find largest contour area and get bounding box for that contour
  > Every image is on low-resolution yet
  > ![image](https://user-images.githubusercontent.com/100391059/200583324-5b5f1f5b-f58c-4b7e-94b5-f667c91092f6.png)
  
  - otsu_foreground_patches
  > Move to high-resolution image and find foreground patches using otsu - threshold
  > Search foreground only in ROI bounding box
  > Return stacked patch in shape of (N,C,H,W) and the format is npy
  > ![image](https://user-images.githubusercontent.com/100391059/200584168-2143e732-7ed1-4024-9b96-e3a1dd24092c.png)
 
  - hard_threshold_patch
  > Move to high-resolution image and find foreground patches using fixed value
  > Search foreground only in ROI bounding box
  > Return stacked patch in shape of (N,C,H,W) and the format is npy

  - full_hardthreshold
  > Similar to 'hard_threshold_patch' but searching whole image
  > This is for the cases where ROI bounding box doesn't work well

  - view_patch
  > Save a suplot figure of collected patches

  - process
  > This function is called in multiprocessing.pool.map()
  > User should choose one method of thresholding in this function

## check_files.py
