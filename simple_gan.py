import tensorflow as tf
import numpy as np
import seaborn as sb
import math
sb.set()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_y(x):
        # generating random samples
    # quadratic function
    return 10 + x * x


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def get_y(x):
    return 10 + x * x


def sample_data(n=10000, scale=100):
    data = []

    x = scale * (np.random.random_sample((n,)) - 0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)


def generator(Z, hsize=[16, 16], reuse=False):
    """

    prams:
    ------
    Z - placeholder for random samples
    hsize=array for the number of units in the 2 hidden layers
    reuse = variable which is used for reusing the same layers.
    """
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        # The output of this function is a 2-dimensional vector which
        # corresponds to the dimensions of the real dataset
        out = tf.layers.dense(h2, 2)

    return out


def discriminator(X, hsize=[16, 16], reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        # 3 hidden layers for the Discriminator out of which first 2 layers
        # size we take input.
        h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, 2)
        out = tf.layers.dense(h3, 1)
    return out, h3

X = tf.placeholder(tf.float32, [None, 2])
Z = tf.placeholder(tf.float32, [None, 2])

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
    r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=f_logits, labels=tf.ones_like(f_logits)))

gen_vars = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(
    gen_loss, var_list=gen_vars)  # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(
    disc_loss, var_list=disc_vars)  # D Train step


sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 256
nd_steps = 10
ng_steps = 10


x_plot = sample_data(n=batch_size)

f = open('loss_logs.csv', 'w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')


for i in range(100001):
    X_batch = sample_data(n=batch_size)
    Z_batch = sample_Z(batch_size, 2)
    _, dloss = sess.run([disc_step, disc_loss], feed_dict={
                        X: X_batch, Z: Z_batch})
    _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" %
          (i, dloss, gloss))

    if i % 10 == 0:
        f.write("%d,%f,%f\n" % (i, dloss, gloss))

    if i % 1000 == 0:
        plt.figure()
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
        xax = plt.scatter(x_plot[:, 0], x_plot[:, 1])
        gax = plt.scatter(g_plot[:, 0], g_plot[:, 1])

        plt.legend((xax, gax), ("Real Data", "Generated Data"))
        plt.title('Samples at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig('plots/iterations/iteration_%d.png' % i)
        plt.close()

        plt.figure()
        rrd = plt.scatter(rrep_dstep[:, 0], rrep_dstep[:, 1], alpha=0.5)
        rrg = plt.scatter(rrep_gstep[:, 0], rrep_gstep[:, 1], alpha=0.5)
        grd = plt.scatter(grep_dstep[:, 0], grep_dstep[:, 1], alpha=0.5)
        grg = plt.scatter(grep_gstep[:, 0], grep_gstep[:, 1], alpha=0.5)

        plt.legend((rrd, rrg, grd, grg), ("Real Data Before G step", "Real Data After G step",
                                          "Generated Data Before G step", "Generated Data After G step"))

        plt.title('Transformed Features at Iteration %d' % i)
        plt.tight_layout()
        plt.savefig('plots/features/feature_transform_%d.png' % i)
        plt.close()
        plt.figure()

        rrdc = plt.scatter(np.mean(rrep_dstep[:, 0]), np.mean(
            rrep_dstep[:, 1]), s=100, alpha=0.5)
        rrgc = plt.scatter(np.mean(rrep_gstep[:, 0]), np.mean(
            rrep_gstep[:, 1]), s=100, alpha=0.5)
        grdc = plt.scatter(np.mean(grep_dstep[:, 0]), np.mean(
            grep_dstep[:, 1]), s=100, alpha=0.5)
        grgc = plt.scatter(np.mean(grep_gstep[:, 0]), np.mean(
            grep_gstep[:, 1]), s=100, alpha=0.5)
        plt.legend((rrdc, rrgc, grdc, grgc), ("Real Data Before G step", "Real Data After G step",
                                              "Generated Data Before G step", "Generated Data After G step"))

        plt.title('Centroid of Transformed Features at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig('plots/features/feature_transform_centroid_%d.png'%i)
        plt.close()

        
f.close()