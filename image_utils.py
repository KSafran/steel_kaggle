import matplotlib.image as mpimg
import numpy as np
import pandas as pd

img_id = '0002cc93b.jpg'
reference_img = mpimg.imread(f'data/train/{img_id}')
train_data = pd.read_csv('data/train.csv')

def show_mask(pixels):
    'given an rle mask string return a mask'
    flat_mask = np.zeros(np.product(reference_img.shape[:2])) 
    if pixels is not np.NaN:
        pix = np.array(pixels.split(' ')).reshape(-1, 2).astype('int')
        for pix_id, run in pix:
            flat_mask[pix_id:pix_id + run] = 1
    return flat_mask.reshape(reference_img.shape[:2], order='F')

def id_to_mask(img_id):
    'given an image id return a 4 channel mask, one channel per class'
    blank_mask = np.zeros((reference_img.shape[0], reference_img.shape[1], 4))
    for i in range(4):
        pixels = train_data[train_data.ImageId_ClassId == f'{img_id}_{i+1}'].iloc[0, 1]
        blank_mask[:, :, i] = show_mask(pixels)
    return blank_mask.astype('int')
        
def extract_imageid(imageid_classid):
    return imageid_classid.split('_')[0]

def get_ids(train_data):
    '''extract image ids from train.csv data'''
    return train_data['ImageId_ClassId'].apply(extract_imageid).tolist()
