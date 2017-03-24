# import urllib

# response = urllib.urlopen('http://deeplearning.net/tutorial/code/utlis.py')
# content = response.read()
# target = open('utlis.py', 'w')
# target.write(content)
# target.close()

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image
from util import tile_raster_images


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.images

print trX.shape, trY.shape, teX.shape, teY.shape

input_nodes = 784
hidden_nodes = 500
alpha = 1.0

vb = tf.placeholder('float', [input_nodes])
hb = tf.placeholder('float', [hidden_nodes])
W = tf.placeholder('float', [784, 500])

X = tf.placeholder('float', [None, 784])
_h0 = tf.nn.sigmoid(tf.matmul(X, W) + hb)  # probabilities of hidden nodes
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
w_pos_grad = tf.matmul(tf.transpose(X), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(X)[0])
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

err = tf.reduce_mean(tf.square(X - v1))
cur_w = np.zeros([784, 500], np.float32)
cur_vb = np.zeros([784], np.float32)
curr_hb = np.zeros([500], np.float32)
prv_w = np.zeros([784, 500], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([500], np.float32)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
print sess.run(err, feed_dict={X: trX, W: prv_w, vb: prv_vb, hb:prv_hb})

epochs, batchsize = 5



