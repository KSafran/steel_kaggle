'''baby model similar to UNet
somewhat based on this kernel
uses binary crossentropy but softmax might make more sense
since there aren't overlapping classes
https://www.kaggle.com/xhlulu/severstal-simple-keras-u-net-boilerplate
'''
import pandas as pd
import numpy as np
from keras import models, layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from src.data_generator import DataGenerator
from src.image_utils import get_ids

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

inputs = models.Input((256, 1600, 1))
c1 = layers.Conv2D(16, (3, 3), activation='elu', padding='same')(inputs)
c1 = layers.Conv2D(16, (3, 3), activation='elu', padding='same')(c1)
m1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(m1)
c2 = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(c2)
m2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(m2)
c3 = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(c3)
m3 = layers.MaxPooling2D((2, 2))(c3)

c_ = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(m3)
c_ = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(c_)

u3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c_)
u3 = layers.concatenate([u3, c3])
c4 = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(u3)
c4 = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(c4)

u2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
u2 = layers.concatenate([u2, c2])
c5 = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(u2)
c5 = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(c5)

u1 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c5)
u1 = layers.concatenate([u1, c1])
c6 = layers.Conv2D(16, (3, 3), activation='elu', padding='same')(u1)
c6 = layers.Conv2D(16, (3, 3), activation='elu', padding='same')(c6)

outputs = layers.Conv2D(4, (1, 1), activation='sigmoid')(c6)

model = models.Model(inputs, outputs)
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=[dice_coef])

train_data = pd.read_csv('data/train.csv')
image_ids = get_ids(train_data)
np.random.seed(10)
np.random.shuffle(image_ids)

train_gen = DataGenerator(
    'train',
    image_ids[:40000],
    batch_size=10)

valid_gen = DataGenerator(
    'train',
    image_ids[40000:],
    batch_size=10,
    shuffle=True)

checkpoint = ModelCheckpoint(
    'models/model.h5', 
    monitor='val_dice_coef', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

history = model.fit_generator(
    train_gen,
    validation_data=valid_gen,
    callbacks = [checkpoint],
    epochs = 5)


