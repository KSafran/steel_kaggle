import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

img_id = '0002cc93b.jpg'

img = mpimg.imread(f'data/train/{img_id}')

train_data = pd.read_csv('data/train.csv')
image_class_1_data = train_data[train_data.ImageId_ClassId == f'{img_id}_1']
image_class_2_data = train_data[train_data.ImageId_ClassId == f'{img_id}_2']
image_class_3_data = train_data[train_data.ImageId_ClassId == f'{img_id}_3']
image_class_4_data = train_data[train_data.ImageId_ClassId == f'{img_id}_4']

def show_mask(pixels):
    flat_mask = np.zeros(np.product(img.shape[:2])) 
    if pixels is not np.NaN:
        pix = np.array(pixels.split(' ')).reshape(-1, 2).astype('int')
        for pix_id, run in pix:
            flat_mask[pix_id:pix_id + run] = 1
    return flat_mask.reshape(img.shape[:2], order='F')

pixels = image_class_1_data.EncodedPixels[0]
mask = show_mask(pixels)

plt.figure(1)
plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(mask)
plt.show()
