# The code is shared on SDSC Github
# Aim: start training with the following code

import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# load image vectors and their labels
all_xs = np.loadtxt('../training_data/TrainingData/imgvectors.txt')
all_ys = np.loadtxt('../training_data/TrainingData/labels.txt')
all_ys = np.float32(all_ys)
print("Data loaded !")

# get the number of images and number of pixels in each image
Nimgs,Npixels = all_xs.shape
Nimgs,Nlabels = all_ys.shape

# initialize the Weights in the neural network
Weights = tf.Variable(tf.random.normal([Npixels, Nlabels]),name='weights')
# now the shape of Weights is like this
# number of row: number of pixels in each image (here is 41*41=1681)
# number of columns: number of classes (here is 5)

# initialize the Biases in the neural network
Biases = tf.Variable(tf.zeros([1, Nlabels]) + 0.1,name='biases')
# now the shape of Biases is like this
# number of row: 1 (Biases here is just a 1D vector)
# number of columns: number of classes (here is 5)

def add_layer(inputs, in_size, out_size, activation_function=None,):

    Wx_plus_b = tf.matmul(inputs, Weights) + Biases
    # each row of the first term (tf.matmul(inputs, Weights)) will be added by Biases
    # now the shape of Wx_plus_b is like this
    # number of row: number of pixels in each image (here is 41*41=1681)
    # number of columns: number of classes (here is 5)

    Wx_plus_b = tf.nn.dropout(Wx_plus_b, rate = 1 - keep_prob)
    # the shape of Wx_plus_b remains unchanged
    # keep_prob is used to remove some insignificant parameters but here we do not remove anything

    # activation function will be applied to each case of W*x+b
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs
    # now the shape of outputs is like this
    # number of row: number of pixels in each image (here is 41*41=1681)
    # number of columns: number of classes (here is 5)


def compute_accuracy(v_xs, v_ys):
    global prediction

    # predict the image class using the trained neural network
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})

    # compute the accuracy using functions embedded in tensorflow
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# hold the places for parameters given by users
keep_prob = tf.compat.v1.placeholder(tf.float32)
xs = tf.compat.v1.placeholder(tf.float32, [None, Npixels])
ys = tf.compat.v1.placeholder(tf.float32, [None, Nlabels])

# invoke the add_layer function to predict
prediction = add_layer(xs, Npixels, Nlabels,  activation_function=tf.nn.softmax)

# we have two options here
# since log(prediction) can be invalid if prediction is 0. to avoid this case, we can use option 2
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.math.log(prediction),reduction_indices=[1])) # option 1
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-10,1.0)), reduction_indices=[1]))   # option 2

# define each training step
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initialize the processing
# important！ Don't forget !
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# create a folder to store the trained model
if os.path.exists("model")==False:
    os.mkdir("model")

# to be unbiased, test data should be different from traning data
# so here we load another dataset to test the trained model
batch_xs = np.loadtxt('../test_data/TestData/imgvectors.txt')
batch_ys = np.loadtxt('../test_data/TestData/labels.txt')
batch_ys = np.float32(batch_ys)

Niter = 500 # number of trainings
starttime = time.process_time()
for i in range(Niter):

    # train the neural network once
    # use training data
    sess.run(train_step, feed_dict={xs: all_xs, ys: all_ys, keep_prob: 1})

    # set a time point to print the current accuracy
    if i % 100 == 0:
        print('step',i,'. Accuracy =',compute_accuracy(batch_xs, batch_ys)) # use test data

    # when the training comes to the last round
    if i == (Niter-1):
##         print the accuracy in the last training below
#        Wx_plus_b = tf.matmul(np.float32(batch_xs), Weights) + Biasesi
#        Wx_plus_b = tf.nn.dropout(Wx_plus_b, 1)
#        outputs = tf.nn.softmax(Wx_plus_b,)
#        correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(batch_ys,1))
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # max accuracy: ~0.84
#        print("Accuracy in the last iteration: " + str(sess.run(accuracy)))
##         print the accuracy in the last trainin above

        # when the training comes to the last round, save the trained model
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, "model/save_net.ckpt")
endtime = time.process_time()
print("Time used: " + str(endtime-starttime) + " seconds.")












