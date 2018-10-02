from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

z_dim = 100


def generator(img_shape, z_dim):
    model = Sequential()

    # Hidden layer
    model.add(Dense(128, input_dim=z_dim))

    # leaky ReLu
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with tanh activation
    model.add(Dense(28*28*1, activation='tanh'))
    model.add(Reshape(img_shape))

    z = Input(shape=(z_dim,))
    img = model(z)


    return Model(z, img)


def discriminator(img_shape):
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))

    # Hidden layer
    model.add(Dense(128))

    # Leaky ReLU
    model.add(LeakyReLU(alpha=0.01))
    # Output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    prediction = model(img)

    return Model(img, prediction)

# Build and compile the discriminator
discriminator = discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(), metrics=['accuracy'])
# build the generator
generator = generator(img_shape, z_dim)

# Generated Image to be used as input
z = Input(shape=(100,))
img = generator(z)

# Keep Discriminator's paramaters constant during generator training
discriminator.trainable = False

# The Discriminator's prediction
prediction = discriminator(img)

# Combined GAN model to train the generator
combined = Model(z, prediction)
combined.compile(loss='binary_crossentropy', optimizer=Adam())

# Training
losses = []
accuracies = []

def train(iterations, batch_size, sample_interval):
	# Load the set
	(X_train, _), (_, _) = mnist.load_data()

	# Rescale -1 to 1
	X_train = X_train / 127.5 - 1.
	X_train = np.expand_dims(X_train, axis=3)

	# Lables for real and fake exmaples
	real = np.ones((batch_size, 1))
	fake = np.zeros((batch_size, 1))

	for iter in range(iterations):
	    # Select a random batch for real images
	    idx = np.random.randint(0, X_train.shape[0], batch_size)
	    imgs = X_train[idx]

	    # Generate a batch of real images
	    z = np.random.normal(0, 1, (batch_size, 100))
	    gen_imgs = generator.predict(z)

	    # Discriminator loss
	    d_loss_real = discriminator.train_on_batch(imgs, real)
	    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
	    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

	    # ---------------------
	    #  Train the Generator
	    # ---------------------
	    z = np.random.normal(0, 1, (batch_size, 100))
	    gen_imgs = generator.predict(z)

	    g_loss = combined.train_on_batch(z, real)

	    if iter % sample_interval == 0:
	    	# Output training progress
	    	print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iter, d_loss[0], 100*d_loss[1], g_loss))

	    	# Save losses and accuracies so they can be plotted after training
	    	losses.append((d_loss[0], g_loss))
	    	accuracies.append(100*d_loss[1])

	    	sample_images(iter)



def sample_images(iteration, image_grid_rows=4, image_grid_columns=4):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale images to 0-1
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(
        4, 4), sharey=True, sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output image grid
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

if __name__ == '__main__':
	iterations = 20000
	batch_size = 128
	sample_interval = 1000
	import warnings; warnings.simplefilter('ignore')

	train(iterations, batch_size, sample_interval)

	losses = np.array(losses)
	# Plot training losses for Discriminator and Generator
	plt.figure(figsize=(10,5))
	plt.plot(losses.T[0], label="Discriminator Loss")
	plt.plot(losses.T[1], label="Generator Loss")
	# plt.savefig('plots/features/feature_transform_%d.png' % i)
	plt.title("Training Losses")
	plt.legend()


	accuracies = np.array(accuracies)



