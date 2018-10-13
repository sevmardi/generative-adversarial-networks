from keras import backend as K
# https://github.com/GANs-in-Action/gans-in-action/blob/master/chapter-7/Chapter_7_SGAN.ipynb

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

