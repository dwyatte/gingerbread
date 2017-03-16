# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
epsilon = 10e-7
learning_rate = 1.0
training_epochs = 50
batch_size = 256
test_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 32
n_input = 784


def dense(input_tensor, input_dim, output_dim, layer_name, act=None):
    """
    simple dense/fully-connected layer
    :param input_tensor:
    :param input_dim:
    :param output_dim:
    :param layer_name:
    :param act:
    :return:
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.get_variable('%s/weights' % layer_name, shape=[input_dim, output_dim],
                                      initializer=tf.contrib.layers.xavier_initializer())
        with tf.name_scope('biases'):
            biases = tf.get_variable('%s/biases' % layer_name, shape=None,
                                     initializer=tf.zeros_initializer(output_dim))
        with tf.name_scope('netinput'):
            netinput = tf.matmul(input_tensor, weights) + biases
        with tf.name_scope('activations'):
            if act:
                activations = act(netinput)
            else:
                activations = netinput
        return activations


# model
with tf.name_scope('input'):
    X = tf.placeholder("float", [None, n_input])
encoded = dense(X, n_input, n_hidden_1, 'encoder1', act=tf.nn.relu)
encoded = dense(encoded, n_hidden_1, n_hidden_2, 'encoder2', act=tf.nn.relu)
encoded = dense(encoded, n_hidden_2, n_hidden_3, 'encoder3', act=tf.nn.relu)
decoded = dense(encoded, n_hidden_3, n_hidden_2, 'decoder1', act=tf.nn.relu)
decoded = dense(decoded, n_hidden_2, n_hidden_1, 'decoder2', act=tf.nn.relu)
decoded = dense(decoded, n_hidden_1, n_input, 'decoder3', act=tf.nn.sigmoid)

# y_true are inputs
y_pred = decoded
y_true = X

# cross entropy loss for each pixel

with tf.name_scope('loss'):
    clipped = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
    logits = tf.log(clipped / (1 - clipped))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, y_true)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    n_train_batch = int(mnist.train.num_examples / batch_size)
    n_test_batch = int(mnist.test.num_examples / batch_size)

    for epoch in range(training_epochs):
        # average train loss
        train_loss = 0.

        for i in range(n_train_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_xs})
            train_loss += l / n_train_batch

            msg = "Epoch %04d:\ttrain loss=%.9f" % (epoch+1, train_loss)

        if epoch % test_step == 0:
            # average test loss
            test_loss = 0.
            for j in range(n_test_batch):
                batch_xs, batch_ys = mnist.test.next_batch(batch_size)
                l = loss.eval(feed_dict={X: batch_xs})
                test_loss += l / n_test_batch
            msg += "\ttest loss=%.9f" % test_loss

        print(msg)

    # Applying encode and decode over test set
    decoded_x = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(decoded_x[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()