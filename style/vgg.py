"""
Based on https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py
"""

import os
import time
import numpy as np
import tensorflow as tf


class VGG16:

    def __init__(self, vgg16_npy_path=None, pool_type=tf.nn.avg_pool):
        if vgg16_npy_path is None:
            path = os.path.dirname(__file__)
            vgg16_npy_path = os.path.join(path, "vgg16.npy")
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.pool_type = pool_type

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3]
        """

        start_time = time.time()
        print("build model started")

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        bgr = tf.concat(axis=3, values=[blue, green, red])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.pool_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.pool_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.pool_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.pool_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.pool_layer(self.conv5_3, 'pool5')

        self.data_dict = None
        print(("build model finished: %fs" % (time.time() - start_time)))

    def pool_layer(self, bottom, name):
        return self.pool_type(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
