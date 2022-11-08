# Digital-Pathology

![image](https://user-images.githubusercontent.com/100391059/200576233-0055a1c9-e329-4103-a73f-4683814a9e10.png)

## main.py
- How to run
``` python
python
```
- Description 
> Training code using MIL (Multiple Instance Learning) for WSI (Whole Slide Image) 
> Use resnet34 as a backbone and put output feature into (multi-head) self-attention transformer encoder. (This part is implemented in MONAI MILmodel)
> Based on MONAI tutorial code for digital pathology

## myfunc.py
- Description
  - Necessary file to run main.py
  
- Functions
  - datalist(foldername)
    - Create a list of dictionary 
    - ex) each dictionary looks like :
    ```
    {
      "image" : '/nfs/thena/shared/thyroid_patch/train/abc.npy'
      "label" : 1.
    }
    ```
    
  - RandPatch(train_patch, data)
    - Get fixed amount of patches randomly. 
    - Default is 20 for training. 
    - In test phase, we use all patches
    
## Preprocessing_mp.py
- How to run

- Description

- Functions
    
