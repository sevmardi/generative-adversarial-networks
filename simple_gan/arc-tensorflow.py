import tensorflow as tf
from distutils.version import LooseVersion
import warnings
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def discriminator(images, reuse=False):
    alpha = 0.2
    with tf.variable_scope('discriminator', reuse=reuse):
        # input : 28x28xn => 14x14x64
        layer = tf.layers.conv2d(images, 64, 5, strides=2, padding='same')
        layer = tf.maximum(alpha * layer, layer)

        # 14x14x64 => 7x7x128
        layer = tf.layers.conv2d(layer, 128, 5, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=True)
        layer = tf.maximum(alpha * layer, layer)

        # 7x7x128 => 4x4x256
        layer = tf.layers.conv2d(layer, 256, 5, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=True)
        layer = tf.maximum(alpha * layer, layer)

        # flatten
        layer = tf.reshape(layer, (-1, 4 * 4 * 256))

        # output layer
        logits = tf.layers.dense(layer, 1, activation=None)
        output = tf.sigmoid(logits)

        return output, logits


def generator(z, out_channel_dim, is_train=True):
    alpha = 0.2
    with tf.variable_scope('generator', reuse=not is_train):

        # input => size of first 'convolutional' layer
        layer = tf.layers.dense(z, 4 * 4 * 256)

        # reshape flat to three dimensional layer (4x4x256)
        layer = tf.reshape(layer, (-1, 4, 4, 256))
        layer = tf.layers.batch_normalization(layer, training=is_train)
        layer = tf.maximum(alpha * layer, layer)

        # 4x4x256 => 7x7x128
        layer = tf.layers.conv2d_transpose(
            layer, 128, 4, strides=1, padding='valid')
        layer = tf.layers.batch_normalization(layer, training=is_train)
        layer = tf.maximum(alpha * layer, layer)

        # 7x7x128 => 14x14x64
        layer = tf.layers.conv2d_transpose(
            layer, 64, 5, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=is_train)
        layer = tf.maximum(alpha * layer, layer)

        # 14x14x64 => 28x28xout_channel_dim
        logits = tf.layers.conv2d_transpose(
            layer, out_channel_dim, 5, strides=2, padding='same')
        output = tf.tanh(logits)

        return output


def model_loss(input_real, input_z, out_channel_dim):
    g_model = generator(input_z, out_channel_dim, is_train=True)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * 0.9))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
        d_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
        g_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
            
    input_real, input_z, lr = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])    
    d_opt, g_opt = model_opt(d_loss, g_loss, lr, beta1)
    
    steps = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                
                steps += 1
        
                batch_images = batch_images * 2
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z, lr: learning_rate})
                _ = sess.run(g_opt, feed_dict={input_z: batch_z, input_real: batch_images, lr: learning_rate})

                if steps % 10 == 0:
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    
                if steps % 100 == 0:
                    show_generator_output(sess, 25, input_z, data_shape[3], data_image_mode)
                    
        if steps % 100 != 0:
            show_generator_output(sess, 25, input_z, data_shape[3], data_image_mode)
            
            
batch_size = 128
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)
