import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, ReLU, Input, Concatenate, ZeroPadding2D
from tensorflow.keras import Model, Sequential
import os
import time
import matplotlib.pyplot as plt
from utils import load_img_train, load_img_test, path, batch_size, buffer_size, img_width, img_height

train_dataset = tf.data.Dataset.list_files(path + '\\train\\*.png')
train_dataset = train_dataset.map(load_img_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

test_dataset = tf.data.Dataset.list_files(path + '\\val\\*.png')
test_dataset = test_dataset.map(load_img_test)
test_dataset = test_dataset.batch(batch_size)

output_channel = 3

def downsample(filters,size,shape,apply_batchnorm = True):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()

    result.add(Conv2D(filters, size, strides=2, padding= 'same', batch_input_shape= shape, kernel_initializer= initializer, use_bias= False))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def upsample(filters, size, shape, apply_dropout = False):

    initializer = tf.random_normal_initializer(0.,0.02)

    result = Sequential()

    result.add(Conv2DTranspose(filters,size,strides=2,padding='same',batch_input_shape=shape,kernel_initializer=initializer, use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:

        result.add(Dropout(0.5))

    result.add(ReLU())

    return result

def build_generator():

    inputs = Input(shape=[256,256,3])

    down_stack = [
        downsample(64,4,(None,256,256,3),apply_batchnorm=False),
        downsample(128,4, (None, 128,128,64)),
        downsample(256, 4, (None,64,64,128)),
        downsample(512,4,(None,32,32,256)),
        downsample(512,4,(None,16,16,512)),
        downsample(512,4,(None,8,8,512)),
        downsample(512,4,(None,4,4,512)),
        downsample(512,4,(None,2,2,512))
    ]

    up_stack = [
        upsample(512,4,(None,1,1,512),apply_dropout=True),
        upsample(512,4,(None,2,2,1024),apply_dropout=True),
        upsample(512,4,(None,4,4,1024),apply_dropout=True),
        upsample(512,4,(None,8,8,1024)),
        upsample(256,4,(None,16,16,1024)),
        upsample(128,4,(None,32,32,512)),
        upsample(64,4,(None,64,64,256))
    ]

    initializer = tf.random_normal_initializer(0.,0.02)

    last = Conv2DTranspose(output_channel,4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')

    x = inputs

    skips = []

    for down in down_stack:

        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip, in zip(up_stack,skips):

        x= up(x)
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs = inputs, outputs = x)

generator = build_generator()
generator.summary()

def downs(filters,size,apply_batch_norm = True):

    initializer = tf.random_normal_initializer(0.,0.02)

    result = Sequential()

    result.add(Conv2D(filters, size, strides=2,padding='same',kernel_initializer=initializer, use_bias=False))

    if apply_batch_norm:

        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def build_discriminator():

    initializer = tf.random_normal_initializer(0.,0.02)

    inp = Input(shape=[256,256,3],name='input_img')
    tar = Input(shape=[256,256,3],name='target_img')

    x = Concatenate()([inp, tar])

    down1 = downs(64,4,False)(x)
    down2 = downs(128,4)(down1)
    down3 = downs(256,4)(down2)

    zero_pad1 = ZeroPadding2D()(down3)

    conv = Conv2D(512,4,strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(batchnorm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)

    last = Conv2D(1,4,strides=1,kernel_initializer=initializer)(zero_pad2)

    return Model(inputs = [inp,tar], outputs = last)

discriminator = build_discriminator()

discriminator.summary()