from __future__ import division, print_function

import argparse
import os
import re
import sys
import tarfile
import pickle
import glob
import random

from six.moves import urllib

import numpy as np
import tensorflow as tf
from utils.nn import conv2d, dense


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIR = '.'
IMAGE_SIZE = 32
NUM_CLASSES = 10
BATCH_SIZE = 128
TRAIN_EPOCHS = 10000
TEST_STEP = 1


def maybe_download_and_extract():
    """
    Download/extract CIFAR-10 if it does not exist
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def yield_batches(is_train, data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """
    Yield batches from CIFAR-10
    """
    def _yield_batches_from(batch_file, batch_size):        
        d = pickle.load(open(batch_file, 'rb'), encoding='latin1')
        data = np.array(d['data'], dtype=float) / 255.0
        data = data.reshape([-1, 3, 32, 32])
        data = data.transpose([0, 2, 3, 1])
        labels = np.array(d['labels'])
        idx = np.array(random.sample(range(data.shape[0]), data.shape[0]))
        for offset in range(0, len(idx), batch_size):
            idx_this_batch = idx[offset:offset+batch_size]
            yield data[idx_this_batch], labels[idx_this_batch]
    if is_train:
        batch_files = glob.glob(os.path.join(DATA_DIR, 'cifar-10-batches-py/data_batch_*'))        
    else:
        batch_files = [os.path.join(DATA_DIR, 'cifar-10-batches-py/test_batch')]        
    for batch_file in batch_files:
        for batch_x, batch_y in _yield_batches_from(batch_file, batch_size):
            yield batch_x, batch_y 


################################################################################
# model
################################################################################
inputs = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
labels = tf.placeholder(tf.int32, [None])
# conv block 1
x = conv2d(inputs, 64, [3, 3], 'conv1', tf.nn.relu)
x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                   padding='SAME', name='pool1')
# conv block 2
x = conv2d(x, 128, [3, 3], 'conv2', tf.nn.relu)
x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                   padding='SAME', name='pool2')
# conv block 3
x = conv2d(x, 128, [3, 3], 'conv3', tf.nn.relu)
x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                   padding='SAME', name='pool3')
# conv block 4
x = conv2d(x, 128, [3, 3], 'conv4', tf.nn.relu)
x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                   padding='SAME', name='pool4')
# conv block 5
x = conv2d(x, 128, [3, 3], 'conv5', tf.nn.relu)
x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                   padding='SAME', name='pool5')
# fc
x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
x = dense(x, 384, 'fc6', tf.nn.relu)
x = dense(x, 192, 'fc7', tf.nn.relu)
logits = dense(x, NUM_CLASSES, 'logits')
# loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# inference
prob = tf.nn.softmax(logits)
init = tf.global_variables_initializer()


################################################################################
# train
################################################################################
maybe_download_and_extract()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(TRAIN_EPOCHS):                
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []        
        print_msg = ''

        for batch_x, batch_y in yield_batches(True):            
            l, p, _ = sess.run([loss, prob, optimizer], 
                               feed_dict={inputs: batch_x, labels: batch_y})            
            train_loss.append(l)
            train_acc.append(np.mean(batch_y == np.argmax(p, axis=1)))            
        batch_loss = np.mean(train_loss)
        batch_acc = np.mean(train_acc)
        print_msg += 'Train: loss=%0.4f, acc=%0.4f' % (batch_loss, batch_acc)
        
        if epoch % TEST_STEP == 0:            
            for batch_x, batch_y in yield_batches(False):
                l, p = sess.run([loss, prob], 
                                feed_dict={inputs: batch_x, labels: batch_y})
                test_loss.append(l)
                test_acc.append(np.mean(batch_y == np.argmax(p, axis=1)))                
            batch_loss = np.mean(test_loss)
            batch_acc = np.mean(test_acc)
            print_msg += '\tTest: loss=%0.4f, acc=%0.4f' % (batch_loss, batch_acc)

        print(print_msg)
