# -*- coding: utf-8 -*-

from __future__ import division, print_function

import tensorflow as tf


def dense(input_tensor, output_dim, layer_name, act=None):
    """
    simple dense/fully-connected layer
    :param input_tensor: input tensor
    :param input_dim: dimensions of input layer
    :param output_dim: dimensions of output layer
    :param layer_name: layer name
    :param act: activation function (linear if none)
    :return:
    """
    input_dim = input_tensor.get_shape()[-1]
    with tf.variable_scope(layer_name):        
        weights = tf.get_variable('weights', shape=(input_dim, output_dim),
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=output_dim,
                                 initializer=tf.zeros_initializer())
        if act:
            return act(tf.matmul(input_tensor, weights) + biases)
        else:
            return tf.matmul(input_tensor, weights) + biases


def conv2d(input_tensor, filters, kernel_size, layer_name, act, 
           strides=[1, 1, 1, 1], padding='SAME'):
    """
    simple conv2d layer
    """
    input_dim = input_tensor.get_shape()[-1]
    with tf.variable_scope(layer_name):        
        kernel = tf.get_variable('weights', 
                                 shape=kernel_size + [input_dim, filters], 
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[filters],
                                 initializer=tf.zeros_initializer())
        if act:
            return act(tf.nn.conv2d(input_tensor, kernel, strides, padding) + biases)
        else:
            return tf.nn.conv2d(input_tensor, kernel, strides, padding) + biases


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
