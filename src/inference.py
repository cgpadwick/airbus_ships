import os
from skimage.data import imread
from skimage.morphology import label
import pandas as pd
import numpy as np
from keras.layers import *
import keras.backend as K
import random
from tqdm import tqdm
import time
from keras.models import load_model



def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


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


def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img[0,:,:,:])
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]



input_dir = '/raid/cgp/data/airbus_ships/'
train_img_dir = '/raid/cgp/data/airbus_ships/train/'
test_img_dir = '/raid/cgp/data/airbus_ships/test/'

test_img_names = [x.split('.')[0] for x in os.listdir(test_img_dir)]

class_model_fname = 'seg_model_ship_classifier_v2.h5'
seg_model_fname = 'seg_model_hypercolumn_ships_only_ioulloss.h5'

class_model = load_model(class_model_fname)
seg_model = load_model(seg_model_fname)

pred_rows = []
for name in tqdm(test_img_names):
    test_img = imread(test_img_dir + name + '.jpg')
    test_img = test_img.reshape(1, 768, 768, 3) / 255.0
    classprob = class_model.predict(test_img)
    if round(classprob[0, 0]) == 0:
        pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': None}]
    else:
        pred_mask = seg_model.predict(test_img)
        pred_mask = pred_mask > 0.3
        rles = multi_rle_encode(pred_mask)
        if len(rles) > 0:
            for rle in rles:
                pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': rle}]
        else:
            pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': None}]

submission_df = pd.DataFrame(pred_rows)[['ImageId', 'EncodedPixels']]
fname = 'submission_' + time.strftime("%Y_%m_%d_%H_%M_%S") + '.csv'
submission_df.to_csv(fname, index=False)



