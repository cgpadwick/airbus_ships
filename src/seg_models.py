from keras.activations import *
from keras.models import *
from keras.layers import *
from keras.utils import multi_gpu_model

import tensorflow as tf

def unet_with_hypercolumn(num_gpus=None, use_dropout=True):

    inputs = Input(shape=(768,768,3))
    conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv0 = BatchNormalization()(conv0)
    conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    comp0 = AveragePooling2D((6,6))(conv0)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(comp0)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    if use_dropout:
        conv1 = Dropout(0.4)(conv1)

    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    if use_dropout:
        conv2 = Dropout(0.4)(conv2)

    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    if use_dropout:
        conv3 = Dropout(0.4)(conv3)

    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    if use_dropout:
        conv4 = Dropout(0.4)(conv4)

    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    upcv6 = UpSampling2D(size=(2,2))(conv5)
    upcv6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv6)
    upcv6 = BatchNormalization()(upcv6)
    mrge6 = concatenate([conv4, upcv6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    upcv7 = UpSampling2D(size=(2,2))(conv6)
    upcv7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv7)
    upcv7 = BatchNormalization()(upcv7)
    mrge7 = concatenate([conv3, upcv7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    upcv8 = UpSampling2D(size=(2,2))(conv7)
    upcv8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv8)
    upcv8 = BatchNormalization()(upcv8)
    mrge8 = concatenate([conv2, upcv8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    upcv9 = UpSampling2D(size=(2,2))(conv8)
    upcv9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv9)
    upcv9 = BatchNormalization()(upcv9)
    mrge9 = concatenate([conv1, upcv9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    dcmp10 = UpSampling2D((6,6), interpolation='bilinear')(conv9)
    mrge10 = concatenate([dcmp10, conv0], axis=3)
    conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=conv11)
    if num_gpus:
        model = multi_gpu_model(model, gpus=num_gpus)
    return model


def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions
        # when the shortcuts go across feature maps of two sizes, they are performed with a
        # stride of 2
        shortcut = \
            Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = add([shortcut, y])
    y = ReLU()(y)

    return y


def upsample_block(y, nb_channels, encoder_connection):

    upcv = UpSampling2D(size=(2, 2))(y)
    upcv = Conv2D(nb_channels, 2, activation='relu', padding='same',
                   kernel_initializer='he_normal')(upcv)
    upcv = BatchNormalization()(upcv)
    upcv = concatenate([encoder_connection, upcv], axis=3)
    return upcv


def unet_with_resnet_encoder(num_gpus=None):

    inputs = Input(shape=(768, 768, 3))

    conv0 = Conv2D(16, 3, strides=(2, 2), activation=None, padding='same',
                   kernel_initializer='he_normal')(inputs)  # 384x384
    conv0 = BatchNormalization()(conv0)
    conv0 = ReLU()(conv0)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0) # 192x192

    # res 2a and 2b
    rb2a = residual_block(pool1, 16)
    rb2b = residual_block(rb2a, 16)

    # res 3a and 3b
    rb3a = residual_block(rb2b, 32, _strides=(2, 2)) # 96x96
    rb3b = residual_block(rb3a, 32)

    # res 4a and 4b
    rb4a = residual_block(rb3b, 64, _strides=(2, 2))  # 48x48
    rb4b = residual_block(rb4a, 64)

    # res 5a and 5b
    rb5a = residual_block(rb4b, 128, _strides=(2, 2))  # 24x24
    rb5b = residual_block(rb5a, 128)

    # Decoder
    upcv1 = upsample_block(rb5b, 64, encoder_connection=rb4b)      # 48x48
    upcv2 = upsample_block(upcv1, 32, encoder_connection=rb3b)     # 96x96
    upcv3 = upsample_block(upcv2, 16, encoder_connection=rb2b)      # 192x192
    upcv4 = upsample_block(upcv3, 8, encoder_connection=conv0)     # 384x384

    upcv5 = UpSampling2D(size=(2, 2))(upcv4)                        # 768x768
    upcv5 = Conv2D(4, 2, activation='relu', padding='same',
                  kernel_initializer='he_normal')(upcv5)
    upcv5 = BatchNormalization()(upcv5)

    output = Conv2D(1, 1, activation='sigmoid')(upcv5)

    model = Model(inputs=inputs, outputs=output)
    if num_gpus:
        model = multi_gpu_model(model, gpus=num_gpus)
    return model
