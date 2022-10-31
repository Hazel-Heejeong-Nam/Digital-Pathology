import numpy as np
import matplotlib.pyplot as plt

bag = np.load('/home/hazel/storage/test/image/CODIPAI-THCB-SS-00873-S-CN-01_fullpatch.npy')
patchnum = bag.shape[0]


plt.figure(figsize = (30,10))
for i in range(320):
    plt.subplot(10,32, i+1)
    img = np.transpose(bag[i].squeeze(),(1,2,0)).astype(np.int32)
    plt.imshow(img)
    plt.axis('off')

plt.savefig('patches/CODIPAI-THCB-SS-00873-S-CN-01_fullpatch_320.png')
