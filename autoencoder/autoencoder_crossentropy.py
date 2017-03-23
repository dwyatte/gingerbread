# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.im import montage
from utils.nn import dense, get_weights


# params
epsilon = 10e-7
learning_rate = 1.0
training_epochs = 100
batch_size = 256
test_step = 1
examples_to_show = 10

# network architecture
n_hidden = [128, 64]
n_input = 784

# other
save_path = 'logs'
save_file = 'autoencoder_mnist_%s' % '_'.join(map(str, n_hidden))


def autoencoder(input_tensor, input_dim, hidden_dims, act=None):
    """
    simple symmetric autoencoder with sigmoidal reconstruction
    :param input_tensor: input tensor (placeholder)
    :param input_dim: dimensions of layer for input tensor
    :param hidden_dims: dimensions of hidden layers
    :param act: activation function of each hidden layer
    :return:
    """
    output_dim = input_dim

    encoded = input_tensor
    for i, hidden_dim in enumerate(hidden_dims):
        encoded = dense(encoded, input_dim, hidden_dim, 'encoder%d' % (i+1), act=act)
        input_dim = hidden_dim

    decoded = encoded
    for i, hidden_dim in enumerate(reversed(hidden_dims[:-1])):
        decoded = dense(decoded, input_dim, hidden_dim, 'decoder%d' % (i+1), act=act)
        input_dim = hidden_dim

    return dense(decoded, hidden_dim, output_dim, 'decoder%d' % (i+2), act=tf.nn.sigmoid)


def cross_entropy(y_pred, y_true):
    """
    cross entropy loss
    :param y_pred:
    :param y_true:
    :return:
    """
    clipped = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
    logits = tf.log(clipped / (1-clipped))
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true)


if __name__ == '__main__':
    # model
    with tf.name_scope('input'):
        X = tf.placeholder('float', [None, n_input])
    y_pred = autoencoder(X, n_input, n_hidden, act=tf.nn.relu)

    # loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(cross_entropy(y_pred, X))

    # optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(save_path, sess.graph)
        sess.run(init)

        mnist = input_data.read_data_sets('MNIST_data')
        n_train_batch = int(mnist.train.num_examples / batch_size)
        n_test_batch = int(mnist.test.num_examples / batch_size)

        for epoch in range(training_epochs):
            # average train loss
            train_loss = 0.

            for i in range(n_train_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, l = sess.run([optimizer, loss], feed_dict={X: batch_xs})
                train_loss += l / n_train_batch

                msg = 'Epoch %04d:\ttrain loss=%.9f' % (epoch+1, train_loss)

            if epoch % test_step == 0:
                # average test loss
                test_loss = 0.
                for j in range(n_test_batch):
                    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
                    l = sess.run(loss, feed_dict={X: batch_xs})
                    test_loss += l / n_test_batch
                msg += '\ttest loss=%.9f' % test_loss

            print(msg)

        saver.save(sess, os.path.join(save_path, save_file), global_step=epoch+1)
        print('Checkpoint saved to %s' % os.path.join(save_path, save_file))

        # Visualize reconstructions
        reconstructions = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
        plt.figure(figsize=(examples_to_show, 2))
        for i in range(examples_to_show):
            plt.subplot(2, examples_to_show, i+1)
            plt.imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='gray')
            plt.subplot(2, examples_to_show, examples_to_show+i+1)
            plt.imshow(np.reshape(reconstructions[i], (28, 28)), cmap='gray')

        # Visualize filters
        plt.figure()
        weights = sess.run(get_weights('encoder1'))
        filters = montage(weights.reshape(1, 28, 28, n_hidden[0]).swapaxes(0, 3),
                          scale=True)
        plt.imshow(np.squeeze(filters), cmap='gray')

        plt.show()
