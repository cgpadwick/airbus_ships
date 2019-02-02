import argparse
import helpers
from keras.models import load_model
import logging
import numpy as np
import os
import pandas as pd
from skimage.data import imread
from tqdm import tqdm


# Need to declare this function here (copied from closure inside helpers.focal_loss)
# since Keras needs to know the definition to restore the model and we can't
# access the fl method inside the closure.
def fl(y_true, y_pred):
    import tensorflow as tf
    from keras import backend as K
    alpha = 0.0
    gamma = 0.0
    eps = 1e-12
    y_pred = K.clip(y_pred, eps, 1. - eps)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + eps)) \
           - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + eps))


def predict(input_dir, model_filename, class_model_filename, output_file, threshold):
    """
    Output predictions on test data, using test time augmentation.
    :param input_dir: path to input directory
    :param model_filename: keras model
    :param output_file: name of output file to be written
    :param threshold: float, confidence value to consider a valid detection
    """

    model = load_model(model_filename, custom_objects={'IoU': helpers.IoU,
                                                       'dice_loss': helpers.dice_loss,
                                                       'focal_loss': helpers.focal_loss,
                                                       'fl': fl,
                                                       'custom_loss': helpers.custom_loss})

    class_model = None
    if class_model_filename:
        class_model = load_model(class_model_filename, custom_objects={'IoU': helpers.IoU,
                                                       'dice_loss': helpers.dice_loss,
                                                       'focal_loss': helpers.focal_loss,
                                                       'fl': fl,
                                                       'custom_loss': helpers.custom_loss})

    test_img_dir = os.path.join(input_dir, 'test')
    test_img_names = [x.split('.')[0] for x in os.listdir(test_img_dir)]

    img_batch = np.zeros((4, 768, 768, 3))

    pred_rows = []
    for name in tqdm(test_img_names):
        test_img = imread(os.path.join(test_img_dir, name + '.jpg')) / 255.0
        img_batch[0, :, :, :] = test_img                # original
        img_batch[1, :, :, :] = test_img[:, ::-1, :]    # flip L/R
        img_batch[2, :, :, :] = test_img[::-1, :, :]    # flip U/D
        img_batch[3, :, :, :] = test_img[::-1, ::-1, :] # flip L/R and U/D

        has_ships = True
        if class_model:
            pred = class_model.predict(img_batch)
            class_pred_prob = (pred[0, :, :, :] +
                         pred[1, :, ::-1, :] +
                         pred[2, ::-1, :, :] +
                         pred[3, ::-1, ::-1, :]) / 4.
            if class_pred_prob >= 0.5:
                has_ships = True
            else:
                has_ships = False

        if has_ships:
            pred = model.predict(img_batch)
            pred_prob = (pred[0, :, :, :] +
                         pred[1, :, ::-1, :] +
                         pred[2, ::-1, :, :] +
                         pred[3, ::-1, ::-1, :]) / 4.
            pred_mask = pred_prob > threshold
            rles = helpers.multi_rle_encode(pred_mask)
            if len(rles) > 0:
                for rle in rles:
                    pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': rle}]
            else:
                pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': None}]
        else:
            pred_rows += [{'ImageId': name + '.jpg', 'EncodedPixels': None}]

    submission_df = pd.DataFrame(pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv(output_file, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        default=None, help="pathname of model to use for prediction")
    parser.add_argument("--class-model", dest='class_model', required=False, type=str,
                        default=None, help="pathname of classification model to use for prediction")
    parser.add_argument("--output-file", dest='output_file', required=False,
                        default='submission.csv', type=str,
                        help="the output file to use for submission")
    parser.add_argument("--input-dir", dest='input_dir', required=False,
                        default='/raid/cgp/data/airbus_ships/',
                        help="input directory location")
    parser.add_argument("--threshold", dest='threshold', required=False, default=0.4, type=float,
                        help="confidence threshold to use for prediction")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    predict(args.input_dir, args.model, args.class_model, args.output_file, args.threshold)

