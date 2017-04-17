# -*- coding: utf-8 -*-

from __future__ import division, print_function

import itertools

import nltk
import unicodecsv as csv
import numpy as np
import tensorflow as tf
from rnn import RNN

nltk.data.path.insert(0, 'nltk_data')

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

vocabulary_size = 8000
learning_rate = 0.01
training_epochs = 20
print_step = 100

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    f.readline()
    reader = csv.reader(f, encoding='utf-8', skipinitialspace=True)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
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

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
n_train = len(X_train)

X = tf.placeholder(tf.int32, None)
y = tf.placeholder(tf.int32, None)

model = RNN(vocabulary_size, hidden_dim=50)
loss = model.calculate_loss(X, y)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        train_loss = 0.

        for i, (t_prev, t) in enumerate(zip(X_train, y_train)):
            _, l = sess.run([optimizer, loss], feed_dict={X: t_prev, y: t})
            train_loss += l / print_step

            if (i+1) % print_step == 0:
                print('Epoch %d:\t%d/%d\tloss=%.9f' % (epoch+1, i+1, n_train, train_loss))
                train_loss = 0.
