import cv2
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import json
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from skimage.data import imread
from skimage.io import imsave
import shutil
import tensorflow as tf
import threading
from tqdm import tqdm
import wandb


def area_isnull(x):
    if x == x:
        return 0
    else:
        return 1


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


def generate_training_and_validation_data(input_dir):

    training_plk_file = os.path.join(input_dir, 'training_df.pkl')
    train_df_plk_file = os.path.join(input_dir, 'train_df.pkl')
    val_pkl_file = os.path.join(input_dir, 'val_df.pkl')

    if not os.path.exists(training_plk_file) or \
            not os.path.exists(val_pkl_file) or \
            not os.path.exists(train_df_plk_file):
        logging.info('pickle files not found, generating')
        train_df = pd.read_csv(input_dir + 'train_ship_segmentations_v2.csv')
        # Remove corrupted image.
        train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']

        train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
        train_df['isnan'].value_counts()
        train_df = train_df.sort_values('isnan', ascending=False)
        # Throw away 100K non-ship images
        train_df = train_df.iloc[100000:]

        train_df['area'] = train_df['EncodedPixels'].apply(calc_area_for_rle)
        train_df.to_pickle(train_df_plk_file)

        train_gp = train_df.groupby('ImageId').sum()
        train_gp = train_gp.reset_index()

        train_gp['class'] = train_gp['area'].apply(calc_class)
        train_gp['class'].value_counts()

        train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())

        # Save the data frames to pickle files so we can restore them later.
        train.to_pickle(training_plk_file)
        val.to_pickle(val_pkl_file)

    logging.info('restoring data frames from pickle files')
    train = pd.read_pickle(training_plk_file)
    val = pd.read_pickle(val_pkl_file)
    train_df = pd.read_pickle(train_df_plk_file)

    # Seed the random number generator with a seed for repeatability.
    random.seed(5432)

    train_isship_list = train['ImageId'][train['isnan']==0].tolist()
    train_isship_list = random.sample(train_isship_list, len(train_isship_list))
    train_nanship_list = train['ImageId'][train['isnan']==1].tolist()
    train_nanship_list = random.sample(train_nanship_list, len(train_nanship_list))
    val_isship_list = val['ImageId'][val['isnan']==0].tolist()
    val_nanship_list = val['ImageId'][val['isnan']==1].tolist()

    res = {'train_isship_list': train_isship_list,
           'train_nanship_list': train_nanship_list,
           'val_isship_list': val_isship_list,
           'val_nanship_list': val_nanship_list,
           'train': train,
           'val': val,
           'train_df': train_df}

    return res


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def iou_measure(y_true, y_pred):
    print(y_true.shape)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return 1. - K.mean( intersection / (union + 1e-6), axis=0)


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean((intersection + eps) / (union + eps), axis=0)


def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    eps = 1e-12
    y_pred=K.clip(y_pred,eps,1.-eps)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + eps)) \
           - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + eps))


def create_data(train_img_dir, image_list, train_df):
    batch_img = []
    batch_mask = []
    for name in image_list:
        tmp_img = imread(train_img_dir + name)
        batch_img.append(tmp_img)
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
        one_mask = np.zeros((768, 768, 1))
        for item in mask_list:
            rle_list = str(item).split()
            tmp_mask = rle_to_mask(rle_list, (768, 768))
            one_mask[:,:,0] += tmp_mask
        batch_mask.append(one_mask)
    img = np.stack(batch_img, axis=0)
    mask = np.stack(batch_mask, axis=0)
    img = img / 255.0
    mask = mask / 255.0
    return img, mask


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def data_generator(isship_list, batch_size, cap_num, train_img_dir, train_df):
    train_img_names_isship = isship_list[:cap_num]
    k = 0
    while True:
        if k + batch_size >= cap_num:
            k = 0
        batch_img_names_is = train_img_names_isship[k:k + batch_size]
        batch_img = []
        batch_mask = []
        for name in batch_img_names_is:
            tmp_img = imread(os.path.join(train_img_dir, name))
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((768, 768, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (768, 768))
                one_mask[:, :, 0] += tmp_mask
            batch_mask.append(one_mask)
        img = np.stack(batch_img, axis=0)
        mask = np.stack(batch_mask, axis=0)
        img = img / 255.0
        mask = mask / 255.0
        k += batch_size
        yield img, mask


def get_keras_callbacks(log_dir):
    reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.7,
                                       patience=10,
                                       verbose=1, mode='auto', cooldown=2,
                                       min_lr=1e-7)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=True, write_images=True)

    callbacks = [reduceLROnPlat, tb_callback]
    return callbacks


def compute(tensor_list):

    y_true = tensor_list[0]
    y_pred = tensor_list[1]
    print(y_true.shape)
    print(y_pred.shape)
    tt = tf.unstack(y_true)
    return K.sum(y_true - y_pred)
    #FP = tf.variable((None, 768, 768, 1), dtype='float32')
    #FP = K.cast(tf.equal(y_true, 0), dtype='float32') + K.cast(tf.greater(y_pred, 0), dtype='float32')
    #FP = tf.logical_and(tf.equal(y_true, 0), tf.greater(y_pred, 0))
    #return K.variable(K.sum(K.cast(FP, dtype='float32')))


    #return tf.confusion_matrix(tf.layers.flatten(y_true), tf.layers.flatten(y_pred))

    #y_true_bin = np.where(y_true != 0, 1, 0).flatten()
    #y_pred_bin = np.where(y_pred > 0, 1, 0).flatten()
    #conf_matrix = confusion_matrix(y_true_bin, y_pred_bin)
    #return K.variable(np.sum(conf_matrix) - np.sum(np.trace(conf_matrix)))


def custom_loss(y_true, y_pred):

    both_tensors = [y_true, y_pred]

    return K.sum(K.square(y_true - y_pred), axis=[1, 2, 3])

    #tt = K.eval(y_true)
    #TN = K.placeholder([None, 768, 768, 1])
    #TN = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    #TN = K.sum(K.variable(TN))
    #return TN



    # this works
    #return K.sum(y_true * y_pred, axis=[1,2,3])

    #return K.sum(y_true - y_pred)
    # This breaks because of arbitrary dimension of y_true
    #tt = tf.unstack(y_true)

    #res = tf.map_fn(compute, (y_true, y_pred))
    #return res

    # try:
    #     y_true_unstack = tf.unstack(y_true)
    #     y_pred_unstack = tf.unstack(y_pred)
    #     res = []
    #     for yt, yp in zip(y_true_unstack, y_pred_unstack):
    #         cm = compute_confusion_matrix(yt, yp)
    #         res.append(np.sum(cm) - np.sum(np.trace(cm)))
    #     return K.variable(res)
    # except Exception as e:
    #     return(K.mean(y_true - y_pred))


def compute_confusion_matrix(y_true, y_pred, threshold=0.5):
    # Binarize the input masks
    y_true_bin = np.where(y_true != 0, 1, 0).flatten()
    y_pred_bin = np.where(y_pred >= threshold, 1, 0).flatten()
    conf_matrix = confusion_matrix(y_true_bin, y_pred_bin)
    return conf_matrix


def compute_score(conf_matrix):

    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    f1  = 2. * precision * recall / (precision + recall)
    return precision, recall, f1


# generate validation image predictions
def output_val_predictions(val_dir, val_list, model, train_df, train_img_dir, num_logged_images=30):

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    logging.info('outputing validation image predictions')

    thres_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    #thres_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    conf_matrices = [[] for x in range(len(thres_range))]

    images = []
    gt = []
    pred = []
    for i in tqdm(range(len(val_list))):
        img_name = os.path.join(train_img_dir, val_list[i])
        img = imread(img_name)
        orig_img = img.copy()
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == val_list[i]].tolist()
        one_mask = np.zeros((768, 768, 1))
        for item in mask_list:
            rle_list = str(item).split()
            tmp_mask = rle_to_mask(rle_list, (768, 768))
            one_mask[:, :, 0] += tmp_mask
        img = img / 255.
        img = img.reshape(1, 768, 768, 3)
        pred_mask = model.predict(img)
        pred_mask = pred_mask.reshape(768, 768, 1)
        input_img = img.reshape(768, 768, 3) * 255.
        rows, cols, channels = input_img.shape
        for j in range(channels):
            idx = input_img[:, :, j] > 255.
            input_img[idx, j] = 255.

        gt_mask = one_mask.astype(np.uint8)
        gt_mask = gt_mask.reshape(768, 768)
        pred_mask = pred_mask.reshape(768, 768)

        for idx, thresh in enumerate(thres_range):
            one_img_conf_matrix = compute_confusion_matrix(gt_mask, pred_mask, threshold=thresh)
            conf_matrices[idx].append(one_img_conf_matrix)

        pred_mask = (pred_mask * 255.0).astype(np.uint8)

        base, ext = os.path.splitext(os.path.basename(img_name))
        out_img = os.path.join(val_dir, os.path.basename(img_name))
        cv2.imwrite(out_img, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        out_gt_mask = os.path.join(val_dir, base + '_gt.png')
        cv2.imwrite(out_gt_mask, gt_mask)
        out_pred_mask = os.path.join(val_dir, base + '_pred.png')
        cv2.imwrite(out_pred_mask, pred_mask)
        if len(images) < num_logged_images * 3:
            images.append(wandb.Image(cv2.resize(orig_img, (256, 256)), caption='Img'))
            images.append(wandb.Image(cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST), caption='GT'))
            images.append(wandb.Image(cv2.resize(pred_mask, (256, 256), interpolation=cv2.INTER_NEAREST), caption='PRED'))

    wandb.log({'examples': images}, commit=True)

    pr_results = {}
    sweep_matrices = [[] for x in range(len(thres_range))]
    prec_bg = []
    recall_bg = []
    fscore_bg = []
    prec_fg = []
    recall_fg = []
    fscore_fg = []
    for idx, thresh in enumerate(thres_range):
        sweep_matrices[idx] = sum(conf_matrices[idx])
        print('threshold level: ' + str(thresh))
        print(sweep_matrices[idx])
        p, r, f1 = compute_score(sweep_matrices[idx])
        prec_bg.append(p[0])
        recall_bg.append(r[0])
        fscore_bg.append(f1[0])
        prec_fg.append(p[1])
        recall_fg.append(r[1])
        fscore_fg.append(f1[1])
        print(p)
        print(r)
        print(f1)
        print('\n\n')
        pr_results[str(thresh)] = {'conf_matrix': sweep_matrices[idx].tolist(),
                              'precision': p.tolist(),
                              'recall': r.tolist(),
                              'f1': f1.tolist()}

    myplots = []
    plt.figure(1)
    plt.plot(thres_range, prec_bg, '-r')
    plt.plot(thres_range, recall_bg, '-b')
    plt.plot(thres_range, fscore_bg, '-g')
    plt.legend(['prec_bg', 'recall_bg', 'fscore_bg'])
    plt.xlabel('threshold')
    plt.suptitle('Background PR-Curve')
    myplots.append(wandb.Image(plt, caption='background'))

    plt.figure(2)
    plt.plot(thres_range, prec_fg, '-r')
    plt.plot(thres_range, recall_fg, '-b')
    plt.plot(thres_range, fscore_fg, '-g')
    plt.legend(['prec_fg', 'recall_fg', 'fscore_fg'])
    plt.xlabel('threshold')
    plt.suptitle('Foreground PR-Curve')
    myplots.append(wandb.Image(plt, caption='foreground'))

    fg_maxidx = np.argmax(fscore_fg)
    bg_maxidx = np.argmax(fscore_bg)

    summary = {'max_prec_fg': prec_fg[fg_maxidx],
               'max_recall_fg': recall_fg[fg_maxidx],
               'max_fscore_fg': fscore_fg[fg_maxidx],
               'max_prec_bg': prec_bg[bg_maxidx],
               'max_recall_bg': recall_bg[bg_maxidx],
               'max_fscore_bg': fscore_bg[bg_maxidx]}
    wandb.log(summary)

    wandb.log({'pr_curves': myplots})

    with open(os.path.join(val_dir, 'pr_results.json'), 'w') as f:
        json.dump(pr_results, f, indent=4)




