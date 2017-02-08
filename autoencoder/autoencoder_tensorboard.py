# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
test_step = 1
examples_to_show = 10
summaries_dir = '/tmp/autoencoder_mnist'

# Network Parameters
n_hidden_1 = 32 # 1st layer num features
n_hidden_2 = 16 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)


def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape))


def bias_variable(shape):
    return tf.Variable(tf.random_normal(shape))


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = nn_layer(x, n_input, n_hidden_1, 'encoder_h1', act=tf.nn.sigmoid)
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = nn_layer(layer_1, n_hidden_1, n_hidden_2, 'encoder_h2', act=tf.nn.sigmoid)

    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = nn_layer(x, n_hidden_2, n_hidden_1, 'decoder_h1', act=tf.nn.sigmoid)
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = nn_layer(layer_1, n_hidden_1, n_input, 'decoder_h2', act=tf.nn.sigmoid)

    return layer_2


# tf Graph input (only pictures)
with tf.name_scope('input'):
    X = tf.placeholder("float", [None, n_input])

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
tf.summary.scalar('cost', cost)

with tf.name_scope('opt'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Merge all the summaries
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    # test_writer = tf.summary.FileWriter(summaries_dir + '/test')

    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            summary, o, c = sess.run([merged, optimizer, cost], feed_dict={X: batch_xs})
        log = "Epoch %04d:\ttrain cost=%.9f" % (epoch + 1, c)
        train_writer.add_summary(summary, epoch)

        # Display logs per epoch step
        if epoch % test_step == 0:
            batch_xs = mnist.test.images[:batch_size]
            c = cost.eval(feed_dict={X: batch_xs})
            log += "\ttest cost=%.9f" % c

        print(log)

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
