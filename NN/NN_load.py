# The code is shared on SDSC Github
# Aim: load the trained model and check accuracy
# This script loads the model trained by NN_train.py only
import tensorflow as tf
import numpy as np
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# load the test data
batch_xs = np.loadtxt('../test_data/TestData/imgvectors.txt')
batch_ys = np.loadtxt('../test_data/TestData/labels.txt')
batch_ys = np.float32(batch_ys)

# get the basic information of the test data
Nimgs,Npixels = batch_xs.shape
Nimgs,Nlabels = batch_ys.shape
print("Data loaded !")

# define and intialize the weights and biases
Weights = tf.Variable(tf.random.normal([Npixels, Nlabels]),name='weights')
Biases = tf.Variable(tf.zeros([1, Nlabels]) + 0.1,name='biases')

# load the trained neural network
saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
    saver.restore(sess, "model/save_net.ckpt")
    Wx_plus_b = tf.matmul(np.float32(batch_xs), Weights) + Biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, rate = 0)
    outputs = tf.nn.softmax(Wx_plus_b,)
    correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(batch_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Current accuracy in the trained model: " + str(sess.run(accuracy)))










