from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#input image dimensions
img_rows, img_cols = 32, 32

#data - shuffled and split between train and test set
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

# Only look at cats [=3] and dogs [=5]
train_picks = np.ravel(np.logical_or(y_train==3,y_train==5)) 
test_picks = np.ravel(np.logical_or(y_test==3,y_test==5))
y_train = np.array(y_train[train_picks]==5,dtype=int)
y_test = np.array(y_test[test_picks]==5,dtype=int)
x_train = x_train[train_picks]
y_train = x_test[test_picks]


if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
	input_shape = (3,img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
	input_shape = (img_rows,img_cols,3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(np.ravel(y_train), num_classes)
y_test = keras.utils.to_categorical(np.ravel(y_test), num_classes)

images = range(0,9)
for i in images:
	plt.subplot(330 + 1 + i)
	plt.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))

plt.show()

