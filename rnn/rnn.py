# -*- coding: utf-8 -*-

from __future__ import division, print_function
import tensorflow as tf


EPSILON = 10e-7


class RNN(object):

    def __init__(self, word_dim, hidden_dim=100):
        """
        Simple RNN
        :param word_dim:
        :param hidden_dim:
        """

        # params
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        # graph
        self.U = tf.get_variable('U', shape=(word_dim, hidden_dim),
                                 initializer=tf.random_uniform_initializer(-tf.sqrt(1. / word_dim),
                                                                           tf.sqrt(1. / word_dim)))
        self.W = tf.get_variable('W', shape=(hidden_dim, hidden_dim),
                                 initializer=tf.random_uniform_initializer(-tf.sqrt(1. / hidden_dim),
                                                                           tf.sqrt(1. / hidden_dim)))
        self.V = tf.get_variable('V', shape=(hidden_dim, word_dim),
                                 initializer=tf.random_uniform_initializer(-tf.sqrt(1. / hidden_dim),
                                                                           tf.sqrt(1. / hidden_dim)))

    def _forward_prop_step(self, s_t_prev, x_t):
        """
        A single step of the unrolled sequence. Note that we are indxing U by x_t.
        This is the same as multiplying U with a one-hot vector.
        :param s_t_prev:
        :param x_t:
        :return:
        """
        return tf.nn.tanh(self.U[x_t, :] + tf.matmul(s_t_prev, self.W))

    def _forward_propagation(self, X):
        """
        Forward propagate with scan operation. Note that scan will insert a singleton dimension along axis=1
        when operating on a vector, so we squeeze it out
        :param X:
        :return:
        """
        s = tf.scan(self._forward_prop_step, X, initializer=tf.zeros(shape=(1, self.hidden_dim)))
        s = tf.squeeze(s)
        return tf.nn.softmax(tf.matmul(s, self.V))

    def predict(self, X):
        """
        Predict label of each word (i.e., argmax)
        :param X:
        :return:
        """
        return tf.argmax(self._forward_propagation(X), axis=0)

    def calculate_loss(self, X, y):
        """
        Mean cross entropy over the sentence. X is variable length, so we need to take the mean here
        :param X:
        :param y:
        :return:
        """
        y_pred = self._forward_propagation(X)
        clipped = tf.clip_by_value(y_pred, EPSILON, 1-EPSILON)
        logits = tf.log(clipped / (1-clipped))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        return tf.reduce_mean(cross_entropy)
