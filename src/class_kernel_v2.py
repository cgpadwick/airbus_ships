#!/usr/bin/env python
# coding: utf-8

# V17: add TTA

# # load packages

# In[1]:


import os
from skimage.data import imread
from skimage.morphology import label
import pandas as pd
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

input_dir = '/raid/cgp/data/airbus_ships/'
train_img_dir = '/raid/cgp/data/airbus_ships/train/'
test_img_dir = '/raid/cgp/data/airbus_ships/test/'

train_df = pd.read_csv(input_dir+'train_ship_segmentations_v2.csv')
train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']

def area_isnull(x):
    if x == x:
        return 0
    else:
        return 1

train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)

train_df = train_df.sort_values('isnan', ascending=False)
train_df = train_df.iloc[100000:]


def rle_to_mask(rle_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask


def calc_area_for_rle(rle_str):
    rle_list = [int(x) if x.isdigit() else x for x in str(rle_str).split()]
    if len(rle_list) == 1:
        return 0
    else:
        area = np.sum(rle_list[1::2])
        return area


train_df['area'] = train_df['EncodedPixels'].apply(calc_area_for_rle)
train_df_isship = train_df[train_df['area'] > 0]
train_df_smallarea = train_df_isship['area'][train_df_isship['area'] < 10]
train_gp = train_df.groupby('ImageId').sum()
train_gp = train_gp.reset_index()

def calc_class(area):
    area = area / (768*768)
    if area == 0:
        return 0
    elif area < 0.005:
        return 1
    elif area < 0.015:
        return 2
    elif area < 0.025:
        return 3
    elif area < 0.035:
        return 4
    elif area < 0.045:
        return 5
    else:
        return 6


train_gp['class'] = train_gp['area'].apply(calc_class)
train_gp['class'].value_counts()

train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())

train_isship_list = train['ImageId'][train['isnan']==0].tolist()
train_isship_list = random.sample(train_isship_list, len(train_isship_list))
train_nanship_list = train['ImageId'][train['isnan']==1].tolist()
train_nanship_list = random.sample(train_nanship_list, len(train_nanship_list))

val_isship_list = val['ImageId'][val['isnan']==0].tolist()
val_nanship_list = val['ImageId'][val['isnan']==1].tolist()

from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  brightness_range = [0.7, 1.3],
                  fill_mode = 'reflect',
                   data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)


def mygenerator(isship_list, nanship_list, batch_size, cap_num):
    train_img_names_nanship = nanship_list[:cap_num]
    train_img_names_isship = isship_list[:cap_num]
    k = 0
    while True:
        if k+batch_size//2 >= cap_num:
            k = 0
        batch_img_names_nan = train_img_names_nanship[k:k+batch_size//2]
        batch_img_names_is = train_img_names_isship[k:k+batch_size//2]
        batch_img = []
        batch_mask = []
        for name in batch_img_names_nan:
            tmp_img = imread(train_img_dir + name)
            batch_img.append(tmp_img)
            batch_mask.append(0)
        for name in batch_img_names_is:
            tmp_img = imread(train_img_dir + name)
            batch_img.append(tmp_img)
            batch_mask.append(1)
        img = np.stack(batch_img, axis=0)
        mask = np.stack(batch_mask, axis=0)

        g_x = image_gen.flow(img, mask,
                             batch_size = img.shape[0], 
                             shuffle=True,
                             seed=None)

        imgaug, maskaug = next(g_x)
        k += batch_size//2
        yield imgaug / 255.0, maskaug

BATCH_SIZE = 4
CAP_NUM = min(len(train_isship_list),len(train_nanship_list))
datagen = mygenerator(train_isship_list, train_nanship_list, batch_size=BATCH_SIZE, cap_num=CAP_NUM)
valgen = mygenerator(val_isship_list, val_nanship_list, batch_size=50, cap_num=CAP_NUM)

numvalimages = 50
val_x, val_y = next(valgen)

inputs = Input(shape=(768,768,3))

c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)  # 384x384

c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)  # 192x192

c3 = Conv2D(128, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(128, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)  # 96x96

c4 = Conv2D(256, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(256, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)  # 48x48

c5 = Conv2D(512, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(512, (3, 3), activation='relu', padding='same') (c5)
p5 = MaxPooling2D(pool_size=(2, 2)) (c5) # 24x24

c6 = Conv2D(1024, (3, 3), activation='relu', padding='same') (p5)
c6 = Conv2D(1024, (3, 3), activation='relu', padding='same') (c6)
p6 = MaxPooling2D(pool_size=(2, 2)) (c6) # 12x12

c7 = Conv2D(1024, (3, 3), activation='relu', padding='same') (p6)
c7 = Conv2D(1024, (3, 3), activation='relu', padding='same') (c7)
p7 = MaxPooling2D(pool_size=(2, 2)) (c7) # 6x6

flatp7 = Flatten() (p7)
d1 = Dense(128, activation='relu') (flatp7)
d1 = Dense(128, activation='relu') (d1)
d2 = Dense(1, activation='sigmoid') (d1)

model = Model(inputs=[inputs], outputs=[d2])
model.summary()


# In[32]:


from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import math, shutil

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.7, 
                                   patience=10, 
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)


if os.path.exists('./log'):
    shutil.rmtree('./log')


tb_callback = TensorBoard(log_dir='./log', histogram_freq=0,  
          write_graph=True, write_images=True)

callbacks_list = [tb_callback, reduceLROnPlat]

NUM_EPOCHS = 100

model.compile(optimizer=Adam(1e-4, decay=0.0), loss='binary_crossentropy', metrics=['acc'])


# # training

# In[33]:


history = model.fit_generator(datagen, steps_per_epoch = 250, epochs = NUM_EPOCHS, callbacks=callbacks_list,
                             validation_data=(val_x, val_y))

model.save('seg_model_ship_classifier_resnet50.h5')

val_list = val['ImageId'].tolist()
train_list = train['ImageId'].tolist()


# In[38]:

from scipy.misc import imresize

def create_data(image_list):
    batch_img = []
    batch_mask = []
    for name in image_list:
        tmp_img = imread(train_img_dir + name)
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
        one_mask = np.zeros((768, 768, 1))
        for item in mask_list:
            rle_list = str(item).split()
            tmp_mask = rle_to_mask(rle_list, (768, 768))
            one_mask[:,:,0] += tmp_mask
        if np.any(one_mask):
            batch_mask.append(1)
        else:
            batch_mask.append(0)
        batch_img.append(imresize(tmp_img, (224, 224, 3)))
    img = np.stack(batch_img, axis=0)
    mask = np.stack(batch_mask, axis=0)
    img = img / 255.0
    return img, mask

from tqdm import tqdm
image_list = val_isship_list
num_actual_images_with_ships = len(image_list)
cnt = 0
for i in tqdm(range(len(image_list))):
    img = imread(train_img_dir + image_list[i])
    input_img, gt_mask = create_data([image_list[i]])
    pred_mask = model.predict(input_img)
    cnt += round(pred_mask[0, 0])
    
acc = float(cnt) / float(num_actual_images_with_ships) * 100
print(acc)
    


# # predict test set and submission with Test Time Augmentation

# In[ ]:


test_img_names = [x.split('.')[0] for x in os.listdir(test_img_dir)]


# In[ ]:


def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img[0,:,:,:])
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


# pred_rows = []
# for name in tqdm(test_img_names):
#     test_img = imread(test_img_dir + name + '.jpg')
#     test_img_1 = test_img.reshape(1,768,768,3)/255.0
#     test_img_2 = test_img_1[:, :, ::-1, :]
#     test_img_3 = test_img_1[:, ::-1, :, :]
#     test_img_4 = test_img_1[:, ::-1, ::-1, :]
#     pred_prob_1 = model.predict(test_img_1)
#     pred_prob_2 = model.predict(test_img_2)
#     pred_prob_3 = model.predict(test_img_3)
#     pred_prob_4 = model.predict(test_img_4)
#     pred_prob = (pred_prob_1 + pred_prob_2[:, :, ::-1, :] + pred_prob_3[:, ::-1, :, :] + pred_prob_4[:, ::-1, ::-1, :])/4
#     pred_mask = pred_prob > opt_threshold
#     rles = multi_rle_encode(pred_mask)
#     if len(rles)>0:
#         for rle in rles:
#             pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': rle}]
#     else:
#         pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': None}]
#
#
# # In[ ]:
#
#
# submission_df = pd.DataFrame(pred_rows)[['ImageId', 'EncodedPixels']]
# submission_df.to_csv('submission.csv', index=False)

