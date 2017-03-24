# -*- coding: utf-8 -*-

from __future__ import division, print_function

import csv
import itertools

import nltk
import numpy as np
import tensorflow as tf

nltk.data.path.insert(0, 'nltk_data')

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

vocabulary_size = 8000
learning_rate = 0.01
training_epochs = 20
batch_size = 100
num_units = 50

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data. We need to keep track of the max sequence length for padding
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
l_train = np.asarray(map(len, X_train))
max_l = max(l_train)

X = tf.placeholder(tf.int32, [None, max_l])
y = tf.placeholder(tf.int32, [None, max_l])
mask = tf.placeholder(tf.float32, [None, max_l])
seqlen = tf.placeholder(tf.int32, [None])

# rnn
# * use embedding to do one-hot lookup
# * outputs of hidden state are returned, so we need to multiply by output weights ourselves
embedding = tf.get_variable('embedding', shape=(vocabulary_size, num_units))
inputs = tf.nn.embedding_lookup(embedding, X)
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)
outputs, states = tf.nn.dynamic_rnn(
    cell=cell,
    sequence_length=seqlen,
    inputs=inputs,
    dtype=tf.float32
)
output_weights = tf.get_variable('output_weights', shape=(num_units, vocabulary_size),
                                 initializer=tf.random_normal_initializer())

# Variable length sequences are a bit difficult in tensorflow. This was helpful:
# https://danijar.com/variable-sequence-lengths-in-tensorflow/

# 1.) we reshape the outputs so that we can just multiply every step by our weights
outputs = tf.reshape(outputs, [-1, num_units])
logits = tf.matmul(outputs, output_weights)
logits = tf.reshape(logits, [-1, max_l, vocabulary_size])

# 2.) we need to mask off just steps we actually used
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
cross_entropy = tf.multiply(cross_entropy, mask)
cross_entropy = tf.reduce_sum(cross_entropy, axis=1) / tf.reduce_sum(mask, axis=1)

loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):

        for i in range(0, len(X_train), batch_size):
            # Get lengths of each row of data
            batch_seqlen = l_train[i:i+batch_size]

            # Mask of valid places in each obs
            batch_mask = np.arange(max_l) < batch_seqlen[:, None]

            # Setup output array and put elements from data into masked positions, convert to one-hot
            batch_x = np.zeros(batch_mask.shape)
            batch_x[batch_mask] = np.hstack((X_train[i:i+batch_size]))

            batch_y = np.zeros(batch_mask.shape)
            batch_y[batch_mask] = np.hstack((y_train[i:i+batch_size]))

            _, l = sess.run([optimizer, loss],
                            feed_dict={X: batch_x, y: batch_y, mask: batch_mask, seqlen: batch_seqlen})

            print('Epoch %04d:\t%04d/%04d\tloss=%.9f' % (epoch+1, i+1, len(X_train), l))
