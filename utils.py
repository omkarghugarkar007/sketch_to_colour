import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt

path = "../path/to/dataset"

buffer_size = 14224
batch_size = 4
img_width = 256
img_height = 256

def load(img_file):

    image = tf.io.read_file(img_file)
    image = tf.image.decode_png(image)

    w = tf.shape(image)[1]

    w = w // 2

    real_img = image[:,:w,:]
    input_img = image[:,w:,:]

    input_img = tf.cast(input_img,tf.float32)
    real_img = tf.cast(real_img, tf.float32)

    return input_img, real_img

def resize(input_img, real_image, height, width):

    input_img = tf.image.resize(input_img,[height,width],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height,width], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_img, real_image

def random_crop(input_img, real_img):

    stacked_img = tf.stack([input_img, real_img], axis= 0)
    cropped_img = tf.image.random_crop(stacked_img, size = [2,img_height,img_width,3])

    return cropped_img[0], cropped_img[1]

def normalize(input_img, real_img):

    input_img = (input_img/127.5) - 1
    real_img = (real_img/127.5) - 1

    return  input_img, real_img

@tf.function()
def random_jitter(input_img, real_img):

    input_img, real_img = resize(input_img,real_img, 286, 286)
    input_img, real_img = random_crop(input_img,real_img)

    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.random_flip_left_right(input_img)
        real_img = tf.image.random_flip_left_right(real_img)

    return  input_img, real_img

def load_img_train(img_file):

    input_img, real_img = load(img_file)
    input_img, real_img = random_jitter(input_img,real_img)
    input_img, real_img = normalize(input_img, real_img)

    return input_img, real_img

def load_img_test(img_file):

    input_img, real_img = load(img_file)
    input_img, real_img = resize(input_img, real_img,img_height, img_width)
    input_img, real_img = normalize(input_img, real_img)

    return input_img, real_img

