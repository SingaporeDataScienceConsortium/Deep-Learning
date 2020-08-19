# The code is shared on SDSC Github
# Aim: training using CNN
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# load the 1D image vectors and their labels
all_xs = np.loadtxt('../training_data/TrainingData/imgvectors.txt') # image vectors
# each row of all_xs is 1D vector containing all pixel values of a single image

all_ys = np.loadtxt('../training_data/TrainingData/labels.txt') # image classes
all_ys = np.float32(all_ys)
# each row of all_ys is one-hot vector
# e.g. [0,0,1,0,0] means the image is in class 3
Nimgs,Npixels = all_xs.shape
Nimgs,Nlabels = all_ys.shape
print("Data loaded !")

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

# we leave shape as the input for the following function since we need more
# than 1 layer. weights should be defined differently with different shapes
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# we leave shape as the input for the following function since we need more
# than 1 layer. biases should be defined differently with different shapes
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # x: vertically stacked 1D image vectors
    # W: filter
    # stride [1, x_movement, y_movement, 1]
    # must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # if padding='SAME', the size of the convoluted image remains unchanged

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # if padding='SAME', the size of the convoluted image remains unchanged

# define placeholder for inputs to network
xs = tf.compat.v1.placeholder(tf.float32, [None, Npixels])   # 41x41
ys = tf.compat.v1.placeholder(tf.float32, [None, Nlabels])
keep_prob = tf.compat.v1.placeholder(tf.float32)

# for each image, convert each 1D vector back to it original 41*41 size
x_image = tf.reshape(xs, [-1, 41, 41, 1])

# resize the image (because the following max pooling need even dimensions of image)
# make sure that the overall image pattern doesn't lose a lot after resizing
x_image = tf.image.resize(x_image, (28, 28), method=0)

# conv1 layer
W_conv1 = weight_variable([5,5, 1,32]) # filter size 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14x14x32

# conv2 layer
W_conv2 = weight_variable([5,5, 32, 64]) # filter size 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7x7x64

# fully connected layer 1
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# the shape changes from [n_samples, 7, 7, 64] to [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# for each image, it changes from 2D to 1D vector

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, rate = 1 - keep_prob)

# fully connected layer 2
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.math.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# important steps
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# load test data
batch_xs = np.loadtxt('../test_data/TestData/imgvectors.txt')
batch_ys = np.loadtxt('../test_data/TestData/labels.txt')
batch_ys = np.float32(batch_ys)
N = 100
starttime = time.process_time()
for i in range(N):
    subsample = np.random.randint(0,Nimgs,10)
    sess.run(train_step, feed_dict={xs: all_xs[subsample,:], ys: all_ys[subsample,:], keep_prob: 1})
    if i % 5 == 0:
        print('step',i,'/',N,'. Accuracy =',compute_accuracy(batch_xs, batch_ys))
endtime = time.process_time()
print("Time used: " + str(endtime-starttime) + " seconds.")