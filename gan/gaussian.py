from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.nn import dense


TRAINING_EPOCHS = 1200
BATCH_SIZE = 12
LOG_STEP = 10
DATA_MU = 4
DATA_SIGMA = 0.5
GENERATOR_RANGE = 8
HIDDEN_SIZE = 4
LEARNING_RATE = 0.03


def generator(input, h_dim):
    h0 = dense(input, h_dim, 'g0', act=tf.nn.tanh)
    h1 = dense(h0, 1, 'g1', act=None)
    return h1


def discriminator(input, h_dim):
    h0 = dense(input, h_dim * 2, 'd0', act=tf.nn.tanh)
    h1 = dense(h0, 1, 'd1', act=tf.nn.sigmoid)
    return h1


####################################################################################################################
# Define GAN
####################################################################################################################
# Generator
with tf.variable_scope('Generator'):
    z = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1))
    G = generator(z, HIDDEN_SIZE)

# Discriminator.
# We create two copies that share parameters since the same network cannot be used w/ different inputs
with tf.variable_scope('Discriminator') as scope:
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1))
    D1 = discriminator(x, HIDDEN_SIZE)
    scope.reuse_variables()
    D2 = discriminator(G, HIDDEN_SIZE)

# Loss and optimizer for each network (see the original paper for details)
loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))

d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

opt_d = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_d, var_list=d_params)
opt_g = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_g, var_list=g_params)

####################################################################################################################
# Train
####################################################################################################################
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(TRAINING_EPOCHS):
    # update discriminator
    x_sample = np.sort(np.random.normal(DATA_MU, DATA_SIGMA, BATCH_SIZE))
    z_sample = np.linspace(-GENERATOR_RANGE, GENERATOR_RANGE, BATCH_SIZE) + np.random.random(BATCH_SIZE) * 0.01
    ld, _ = sess.run([loss_d, opt_d], {
        x: np.reshape(x_sample, (BATCH_SIZE, 1)),
        z: np.reshape(z_sample, (BATCH_SIZE, 1))
    })

    # update generator
    z_sample = np.linspace(-GENERATOR_RANGE, GENERATOR_RANGE, BATCH_SIZE) + np.random.random(BATCH_SIZE) * 0.01
    lg, _ = sess.run([loss_g, opt_g], {
        z: np.reshape(z_sample, (BATCH_SIZE, 1))
    })

    if epoch % LOG_STEP == 0:
        print('{}: {}\t{}'.format(epoch, ld, lg))

####################################################################################################################
# Plot data distribution, generated samples, decision function
####################################################################################################################
n_points = 10000
n_bins = 100
x_point_domain = np.linspace(-GENERATOR_RANGE, GENERATOR_RANGE, n_points)
x_bin_domain = np.linspace(-GENERATOR_RANGE, GENERATOR_RANGE, n_bins)

# decision function
decision_function = np.zeros((n_points, 1))
for i in range(n_points // BATCH_SIZE):
    decision_function[BATCH_SIZE * i:BATCH_SIZE * (i + 1)] = sess.run(D1, {
        x: np.reshape(x_point_domain[BATCH_SIZE * i:BATCH_SIZE * (i + 1)], (BATCH_SIZE, 1))
    })

# data distribution
data = np.sort(np.random.normal(DATA_MU, DATA_SIGMA, n_points))
p_data, _ = np.histogram(data, bins=x_bin_domain, density=True)

# generated samples
generated = np.zeros((n_points, 1))
for i in range(n_points // BATCH_SIZE):
    generated[BATCH_SIZE * i:BATCH_SIZE* (i + 1)] = sess.run(G, {
        z: np.reshape(x_point_domain[BATCH_SIZE * i:BATCH_SIZE * (i + 1)], (BATCH_SIZE, 1))
    })
p_generated, _ = np.histogram(generated, bins=x_bin_domain, density=True)

# these may have changed shape slightly, regenerate
x_point_domain = np.linspace(-GENERATOR_RANGE, GENERATOR_RANGE, len(decision_function))
x_bin_domain = np.linspace(-GENERATOR_RANGE, GENERATOR_RANGE, len(p_data))

plt.plot(x_point_domain, decision_function)
plt.plot(x_bin_domain, p_data)
plt.plot(x_bin_domain, p_generated)
plt.legend(['Decision function', 'Data', 'Generated'])
plt.show()
