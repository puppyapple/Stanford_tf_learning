#%%
# import 
import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
print ("Imported!")

#%%
# load data
notmnist = input_data.read_data_sets('./data', one_hot=True)

#%%
# define variables and placeholders
batch_size = 128
n_epochs = 30
learning_rate = 0.01

X = tf.placeholder(tf.float32,[batch_size, 784], name="X_placeholder")
Y = tf.placeholder(tf.float32,[batch_size, 10], name="Y_placeholder")

w = tf.Variable(tf.random_normal([784, 10] ,stddev=0.01), name="Weights")
b = tf.Variable(tf.zeros([1, 10]), name="Bias")

logits = tf.matmul(X, w) + b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#%%
# training 
print('start training')
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    batch_num = int(notmnist.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss = 0
        for k in range(batch_num):
            x_batch, y_batch = notmnist.train.next_batch(batch_size)
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X: x_batch, Y: y_batch})
            total_loss += batch_loss
        print('Average loss epoch {0}: {1}'.format(i, total_loss/batch_size))
    print('Total time: {0} seconds'.format(time.time() - start_time))
    print('Optimization finished!')
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    batch_num = int(notmnist.test.num_examples/batch_size)
    print(notmnist.test.num_examples)
    print(batch_num*batch_size)
    total_correct_preds = 0
    for i in range(batch_num):
        x_batch, y_batch = notmnist.test.next_batch(batch_size)
        batch_accuracy = sess.run(accuracy, feed_dict={X: x_batch, Y: y_batch})
        total_correct_preds += batch_accuracy
    print('Accuracy {0}'.format(total_correct_preds/notmnist.test.num_examples))