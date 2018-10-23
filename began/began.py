import tensorflow as tf
import numpy as np


self.repeat_num = int(np.log2(height)) - 2.


def generator(z, hidden_num, output_num, repeat_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8,  hidden_num]))
        x = slim.fully_connected(z, num_output, activiation_fn=None)
        x = reshape(x,  8,  8,  hidden_num, data_format)

        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x,  channel_num, 3, 1,
                            activation_fn=tf.nn.elu,  data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1,
                            activation_fn=tf.nn.elu,    data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x,  channel_num,    3,  2,
                                activation_fn=tf.nn.elu,    data_format=data_format)
                #x = tf.    tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
                x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
                z = x = slim.fully_connected(x, z_num, activation_fn=None)

                num_output = int(np.prod([8, 8,  hidden_num]))
                x = slim.fully_connected(x, num_output, activation_fn=None)
                x = reshape(x,  8,  8,  hidden_num, data_format)
                for idx in range(repeat_num):
                    x = slim.conv2d(
                        x,  hidden_num, 3,  1,  activation_fn=tf.nn.elu,    data_format=data_format)
                    x = slim.conv2d(
                        x,  hidden_num, 3,  1, activation_fn=tf.nn.elu,  data_format=data_format)

                    if idx < repeat_num - 1:
                        x = upscale(x,  2,  data_format)
                    out  = slim.conv2d(x, input_channel, 3,1, activation_fn=None,   data_format=data_format)
                    variables = tf.contrib.framework.get_variables(vs)
                    