from Gan_Model import generator, discriminator
from train import checkpoint, checkpoint_dirs, generated_img
import tensorflow as tf
from utils import load_img_test
import numpy as np
import os
import matplotlib.pyplot as plt

generator.load_weights(generator.h5)

for _,_,files in os.walk('test_images'):
    for file in files:
        img_file = os.path.join('test_images',file)
        example_input, example_target = load_img_test(img_file)
        example_target = np.reshape(example_target, (256,256,3))
        example_input = np.reshape(example_input, (1,256,256,3))
        generated_img(generator,example_input,example_target)