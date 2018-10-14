import tensorflow as tf

Y = tf.placeholder(tf.float32, shape=[None, num_labels])

Dhidden = 256  # Hidden units of discriminator's network
Ghideen = 512  # Hidden units of Generator's network
K = 8

# Discirminator network


def discriminator(x, y):
    u = tf.reshape(tf.matmul(x, DW1x) + tf.matmul *
                   (y, DW1y) + Db1, [-1, K, Dhidden])
    Dh1 = tf.nn.dropout(tf.reduce_max(u,	reduction_indices=[1]),	keep_prob)
    return tf.nn.sigmoid(tf.matmul(Dh1,	DW2) + Db2)

#	Generator	Network


def generator(z, y):
    Gh1 = tf.nn.relu(tf.matmul(Z, GW1z) + tf.matmul(Y, GW1y) + Gb1)
    G = tf.nn.sigmoid(tf.matmul(Gh1, GW2) + Gb2)
    return G

G_sample = generator(Z, Y)
DG = discriminator(G_sample, Y)

Dloss = -tf.reduce_mean(tf.log(discriminator(X,	Y)) + tf.log(1 - DG))
Gloss = tf.reduce_mean(tf.log(1 - DG) - tf.log(DG + 1e-9))


X_mb,	y_mb = mnist.train.next_batch(mini_batch_size)
Z_sample = sample_Z(mini_batch_size, noise_dim)

_,	D_loss_curr = sess.run([Doptimizer, Dloss],
                          feed_dict={X:	X_mb,	Z:	Z_sample,	Y: y_mb,
                                     keep_prob: 0.5})

_,	G_loss_curr = sess.run([Goptimizer, Gloss],
                          feed_dict={Z:	Z_sample,	Y: y_mb,	keep_prob: 1.0})

nsamples = 6
Z_sample = sample_Z(nsamples, noise_dim)
y_sample = np.zeros(shape=[nsamples, num_labels])
y_sample[:,	7] = 1  # generating image based on label

samples = sess.run(G_sample, feed_dict={Z: Z_sample, Y: y_sample})


