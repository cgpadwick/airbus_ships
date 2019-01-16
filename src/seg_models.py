from keras.activations import *
from keras.models import *
from keras.layers import *
from keras.utils import multi_gpu_model

import tensorflow as tf

def unet_with_hypercolumn(num_gpus=None, use_dropout=True):
    """
    unet architecture with hypercolumn
    :param num_gpus: int, specify for multi gpu run
    :param use_dropout: bool, specify true to use dropout in the network
    :return: keras model
    """

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
    """
    residual block implementation
    :param y: input layer
    :param nb_channels: int
    :param _strides: tuple, specify strided conv
    :param _project_shortcut: specify shortcut connection
    :return: keras layer
    """
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
    """
    Upsample block
    :param y: input layer
    :param nb_channels: int
    :param encoder_connection: layer to connect to from the encoder
    :return: keras layer
    """

    upcv = UpSampling2D(size=(2, 2))(y)
    upcv = Conv2D(nb_channels, 2, activation='relu', padding='same',
                   kernel_initializer='he_normal')(upcv)
    upcv = BatchNormalization()(upcv)
    upcv = concatenate([encoder_connection, upcv], axis=3)
    return upcv


def unet_with_resnet_encoder(num_gpus=None):
    """
    unet architecture with resnet encoder
    :param num_gpus: integer, specify for multi-gpu run
    :return: keras model
    """

    inputs = Input(shape=(768, 768, 3))

    conv0 = Conv2D(16, 3, strides=(2, 2), activation=None, padding='same',
                   kernel_initializer='he_normal')(inputs)  # 384x384
    conv0 = BatchNormalization()(conv0)
    conv0 = ReLU()(conv0)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0) # 192x192

    # res 2a and 2b, 2c
    rb2a = residual_block(pool1, 16)
    rb2b = residual_block(rb2a, 16)
    rb2c = residual_block(rb2b, 16)

    # res 3a and 3b, 3c
    rb3a = residual_block(rb2c, 32, _strides=(2, 2)) # 96x96
    rb3b = residual_block(rb3a, 32)
    rb3c = residual_block(rb3b, 32)

    # res 4a and 4b, 4c
    rb4a = residual_block(rb3c, 64, _strides=(2, 2))  # 48x48
    rb4b = residual_block(rb4a, 64)
    rb4c = residual_block(rb4b, 64)

    # res 5a and 5b, 5c
    rb5a = residual_block(rb4c, 128, _strides=(2, 2))  # 24x24
    rb5b = residual_block(rb5a, 128)
    rb5c = residual_block(rb5b, 128)

    # res 6a, 6b, 6c
    rb6a = residual_block(rb5c, 256, _strides=(2, 2))  # 12x12
    rb6b = residual_block(rb6a, 256)
    rb6c = residual_block(rb6b, 256)

    # res 7a, 7b, 7c
    rb7a = residual_block(rb6c, 512, _strides=(2, 2))  # 6x6
    rb7b = residual_block(rb7a, 512)
    rb7c = residual_block(rb7b, 512)

    # Decoder
    upcv1 = upsample_block(rb7c, 256, encoder_connection=rb6c)  # 12x12
    upcv2 = upsample_block(upcv1, 128, encoder_connection=rb5c) # 24x24
    upcv3 = upsample_block(upcv2, 64, encoder_connection=rb4c)  # 48x48
    upcv4 = upsample_block(upcv3, 32, encoder_connection=rb3c)  # 96x96
    upcv5 = upsample_block(upcv4, 16, encoder_connection=rb2c)  # 192x192
    upcv6 = upsample_block(upcv5, 8, encoder_connection=conv0)  # 384x384

    upcv7 = UpSampling2D(size=(2, 2))(upcv6)                    # 768x768
    upcv8 = Conv2D(4, 2, activation='relu', padding='same',
                  kernel_initializer='he_normal')(upcv7)
    upcv9 = BatchNormalization()(upcv8)

    output = Conv2D(1, 1, activation='sigmoid')(upcv9)

    model = Model(inputs=inputs, outputs=output)
    if num_gpus:
        model = multi_gpu_model(model, gpus=num_gpus)
    return model


def resnet_classifier(num_gpus=None):
    """
    Resnet classification network.
    :param num_gpus: integer, specify for multi-gpu run
    :return: Keras model
    """

    inputs = Input(shape=(768, 768, 3))

    conv0 = Conv2D(16, 3, strides=(2, 2), activation=None, padding='same',
                   kernel_initializer='he_normal')(inputs)  # 384x384
    conv0 = BatchNormalization()(conv0)
    conv0 = ReLU()(conv0)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0) # 192x192

    # res 2a, 2b, 2c
    rb2a = residual_block(pool1, 16)
    rb2b = residual_block(rb2a, 16)
    rb2c = residual_block(rb2b, 16)

    # res 3a, 3b, 3c
    rb3a = residual_block(rb2c, 32, _strides=(2, 2)) # 96x96
    rb3b = residual_block(rb3a, 32)
    rb3c = residual_block(rb3b, 32)

    # res 4a, 4b, 4c
    rb4a = residual_block(rb3c, 64, _strides=(2, 2))  # 48x48
    rb4b = residual_block(rb4a, 64)
    rb4c = residual_block(rb4b, 64)

    # res 5a, 5b, 5c
    rb5a = residual_block(rb4c, 128, _strides=(2, 2))  # 24x24
    rb5b = residual_block(rb5a, 128)
    rb5c = residual_block(rb5b, 128)

    # res 6a, 6b, 6c
    rb6a = residual_block(rb5c, 256, _strides=(2, 2))  # 12x12
    rb6b = residual_block(rb6a, 256)
    rb6c = residual_block(rb6b, 256)

    # res 7a, 7b, 7c
    rb7a = residual_block(rb6c, 512, _strides=(2, 2))  # 6x6
    rb7b = residual_block(rb7a, 512)
    rb7c = residual_block(rb7b, 512)

    flatp8 = Flatten()(rb7c)
    d1 = Dense(128, activation='relu')(flatp8)
    d2 = Dense(1, activation='sigmoid')(d1)

    model = Model(inputs=inputs, outputs=d2)
    if num_gpus:
        model = multi_gpu_model(model, gpus=num_gpus)
    return model
