"""
adapted from https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from utils.nn import dense


BATCH_SIZE = 128
TRAINING_EPOCHS = 100000
LOG_STEP = 1000
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
Z_SIZE = 100


def discriminator(x, hidden_size):
    h0 = dense(x, hidden_size, 'd0', act=tf.nn.relu)
    h1 = dense(h0, 1, 'd1', act=tf.nn.sigmoid)
    return h1


def generator(z, hidden_size):
    h0 = dense(z, hidden_size, 'g0', act=tf.nn.relu)
    h1 = dense(h0, 784, 'g1', act=tf.nn.sigmoid)
    return h1

with tf.variable_scope('Generator') as scope:
    Z = tf.placeholder(tf.float32, shape=[None, Z_SIZE])
    G_sample = generator(Z, HIDDEN_SIZE)

with tf.variable_scope('Discriminator') as scope:
    X = tf.placeholder(tf.float32, shape=[None, 784])
    D_real = discriminator(X, HIDDEN_SIZE)
    scope.reuse_variables()
    D_fake = discriminator(G_sample, HIDDEN_SIZE)

D_loss = tf.reduce_mean(-tf.log(D_real) - tf.log(1. - D_fake))
G_loss = tf.reduce_mean(-tf.log(D_fake))

G_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
D_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

D_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(D_loss, var_list=D_params)
G_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(G_loss, var_list=G_params)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('logs', sess.graph)

for epoch in range(TRAINING_EPOCHS):

    X_batch, _ = mnist.train.next_batch(BATCH_SIZE)

    _, D_loss_curr = sess.run([D_opt, D_loss], feed_dict={X: X_batch, Z: np.random.uniform(-1., 1., size=(BATCH_SIZE, Z_SIZE))})
    _, G_loss_curr = sess.run([G_opt, G_loss], feed_dict={Z: np.random.uniform(-1., 1., size=(BATCH_SIZE, Z_SIZE))})

    if epoch % LOG_STEP == 0:
        print('{}: {}\t{}'.format(epoch, D_loss_curr, G_loss_curr))

        ################################################################################################################
        # plot
        ################################################################################################################
        if not os.path.exists('out/'):
            os.makedirs('out/')

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        samples = sess.run(G_sample, feed_dict={Z: np.random.uniform(-1., 1., size=(16, Z_SIZE))})
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        plt.savefig('out/{}.png'.format(str(epoch).zfill(7)), bbox_inches='tight')
        plt.close(fig)
