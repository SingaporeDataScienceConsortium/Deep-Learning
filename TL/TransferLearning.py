# The code is shared on SDSC Github
# Aim: use pre-trained vgg16 for image classification
import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

vgg_mean = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle = True).item()

    def build(self, rgb):
        print("Model loading in progress..")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        self.fc6 = self.fc_layer(pool5, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

    def max_pool(self, bottom, name):
        return tf.nn.max_pool2d(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name, reuse=True):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def fc_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

def recognition():
    tf.compat.v1.reset_default_graph()
    img = skimage.io.imread("./UserTestImages/test (1).jpg")
    img = img / 255.0

    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]

    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    resized_img = resized_img.reshape((1, 224, 224, 3))

    with tf.compat.v1.Session() as sess:
        images = tf.compat.v1.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: resized_img}

        vgg = Vgg16(vgg16_npy_path='./vgg16.npy')
        vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        prob = prob[0]

        synset = [l.strip() for l in open('vgg16_labels.txt').readlines()]
        pred = np.argsort(prob)[::-1]

        top1 = synset[int(pred[0])]
        print('===========================================\nPredicted Result')
        print("Top1: ", top1, '\nConfidence Level:',100*prob[pred[0]],'%.')

        print('===========================================\nOther Possible Results')
        for i in range(10):
            print((i+1),'.',synset[int(pred[i+1])])
            print('    Confidence Level:',100*prob[pred[i+1]],'%')

recognition()
