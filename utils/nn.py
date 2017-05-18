# -*- coding: utf-8 -*-

from __future__ import division, print_function

import tensorflow as tf


def dense(input_tensor, output_dim, layer_name, act):
    """
    simple dense/fully-connected layer
    :param input_tensor: input tensor
    :param input_dim: dimensions of input layer
    :param output_dim: dimensions of output layer
    :param layer_name: layer name
    :param act: activation function (linear if none)
    :return:
    """
    with tf.variable_scope(layer_name):
        weights = tf.get_variable('weights', shape=(input_tensor.get_shape()[1], output_dim),
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=output_dim,
                                 initializer=tf.zeros_initializer())
        if act:
            return act(tf.matmul(input_tensor, weights) + biases)
        else:
            return tf.matmul(input_tensor, weights) + biases


def get_weights(layer_name):
    """
    get weights for layer
    :param layer_name: layer name
    :return:
    """
    with tf.variable_scope(layer_name, reuse=True):
        return tf.get_variable('weights')


def get_biases(layer_name):
    """
    get biases for layer
    :param layer_name: layer name
    :return:
    """
    with tf.variable_scope(layer_name, reuse=True):
        return tf.get_variable('biases')
