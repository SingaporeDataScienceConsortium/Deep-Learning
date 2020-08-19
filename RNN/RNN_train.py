# The code is shared on SDSC Github
# Aim: training using RNN
import tensorflow as tf
import time
import random
import numpy as np
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# pre-defined parameters
lr = 0.001
training_iters = 10000
batch_size = 256

n_inputs = 41   # data input (data shape: 41*41)
n_steps = 41    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 5

# tf Graph input
x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (41, 128)
    'in': tf.Variable(tf.random.normal([n_inputs, n_hidden_units])),
    # (128, 5)
    'out': tf.Variable(tf.random.normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (5, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (batch_size * 41 steps, 41 inputs)
    X = tf.reshape(X, [-1, n_inputs]) # deeee?

    # into hidden
    # X_in = (batch_size * 41 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (batch_size, 41 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden_units)

    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    # hidden layer for output as the final results
    #############################################

    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 5)

    return results

pred = RNN(x, weights, biases)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# load training data
training_xs = np.loadtxt('../training_data/TrainingData/imgvectors.txt')
training_xs = np.float32(training_xs)

training_ys = np.loadtxt('../training_data/TrainingData/labels.txt')
training_ys = np.float32(training_ys)

# load test data
test_xs = np.loadtxt('../test_data/TestData/imgvectors.txt')
test_xs = np.float32(test_xs)

test_ys = np.loadtxt('../test_data/TestData/labels.txt')
test_ys = np.float32(test_ys)

starttime = time.process_time()
with tf.compat.v1.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:

        # we only randomly choose a subset (batch_size) of all data for computational efficiency
        rn = [x for x in range(int(np.round(training_ys.shape[0])))]
        random.shuffle(rn)
        trainingbatch_xs = training_xs[rn[0:batch_size],:]
        trainingbatch_ys = training_ys[rn[0:batch_size],:]

        trainingbatch_xs = trainingbatch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: trainingbatch_xs, y: trainingbatch_ys,})
        if step % 5 == 0:
            rn = [x for x in range(int(np.round(test_ys.shape[0])))]
            random.shuffle(rn)
            testbatch_xs = test_xs[rn[0:batch_size],:]
            testbatch_ys = test_ys[rn[0:batch_size],:]
            testbatch_xs = testbatch_xs.reshape([batch_size, n_steps, n_inputs])
            print('step ',step*batch_size/training_iters*100,'% .',
                  ' Accuracy =',sess.run(accuracy, feed_dict={x: testbatch_xs, y: testbatch_ys,}))
        step += 1
endtime = time.process_time()
print(endtime-starttime)