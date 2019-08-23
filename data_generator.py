import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import keras

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
        

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_ids, batch_size, shuffle=True):
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        'Gets batch indices for iteration index then generates data'
        # Get indices for batch
        indicies = self.indicies[index * self.batch_size:(index + 1) * self.batch_size]
        # image ids for this batch
        image_ids_temp = [self.image_ids[k] for k in indicies]

        X, y = self.__data_generation(image_ids_temp)
        return X, y

    def on_epoch_end(self):
        'Randomize order of image ids'
        self.indicies = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indicies)

    def __data_generation(self, image_ids_temp):
        'generates x and y values for batch of ids'
        inputs = np.zeros((self.batch_size, reference_img.shape[0], reference_img.shape[1], reference_img.shape[2]))
        outputs = np.zeros((self.batch_size, reference_img.shape[0], reference_img.shape[1], 4))
        for i, image_id  in enumerate(image_ids_temp):
            inputs[i, :, :, :] = mpimg.imread(f'data/train/{image_id}') / 256
            outputs[i, :, :, :] = id_to_mask(image_id)
        return inputs, outputs

sample_indicies = train_data.ImageId_ClassId[:100].apply(lambda x:x.split('_')[0])

train_dg = DataGenerator(sample_indicies, batch_size=10)

x, y = train_dg.__getitem__(1)
