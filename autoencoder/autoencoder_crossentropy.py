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
learning_rate = 1.0
l1_lambda = 10e-5
training_epochs = 100
batch_size = 256
test_step = 1
examples_to_show = 10
epsilon = 1e-10

# Network Parameters
n_hidden = 32 # 1st layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# glorot initialization
minval = -4 * np.sqrt(6. / (n_hidden + n_input))
maxval = 4 * np.sqrt(6. / (n_hidden + n_input))

weights = {
    'encoder_h1': tf.Variable(tf.random_uniform([n_input, n_hidden], minval, maxval)),
    'decoder_h1': tf.Variable(tf.random_uniform([n_hidden, n_input], minval, maxval)),
}
biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden])),
    'decoder_b1': tf.Variable(tf.zeros([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                biases['encoder_b1']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                            biases['decoder_b1']))
    return layer_1

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer
binary_crossentropy = -(tf.multiply(y_true, tf.log()) +
                        tf.multiply((1-y_true), tf.log(tf.clip_by_value(1-y_pred, epsilon, 1-epsilon))))
l1 = tf.abs(weights['encoder_h1'])
cost = tf.reduce_mean(binary_crossentropy) + l1_lambda*tf.reduce_mean(l1)
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_train_batch = int(mnist.train.num_examples / batch_size)
    total_test_batch = int(mnist.test.num_examples / batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        avg_train_cost = 0.

        # Loop over all batches
        for i in range(total_train_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            avg_train_cost += c / total_train_batch

        log = "Epoch %04d:\ttrain cost=%.9f" % (epoch+1, avg_train_cost)

        # Display train/test cost per test_step
        if epoch % test_step == 0:
            avg_test_cost = 0.
            for j in range(total_test_batch):
                batch_xs, batch_ys = mnist.test.next_batch(batch_size)
                c = cost.eval(feed_dict={X: batch_xs})
                avg_test_cost += c / total_test_batch
            log += "\ttest cost=%.9f" % avg_test_cost

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
