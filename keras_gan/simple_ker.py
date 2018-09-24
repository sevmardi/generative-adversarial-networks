import keras
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from tqdm import tqdm
import numpy as np


np.random.seed(1000)

random_dim = 100

def load_mnist():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = (x_train.astype(np.float32) - 127.5)/ 127.5
	x_train =  x_train.reshape(60000, 784)
	return (x_train, y_train, x_test, y_test)

