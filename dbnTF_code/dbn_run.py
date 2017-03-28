# Getting the MNIST data provided by Tensorflow
from tensorflow.examples.tutorials.mnist import input_data
from NN import NN
from rbm import RBM
import numpy as np

# Loading in the mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, \
                     mnist.test.labels

# print trX.shape
# exit()

RBM_hidden_sizes = [128]  # create 2 layers of RBM with size 400 and 100

# Since we are training, set input as training data
inpX = trX

# Create list to hold our RBMs
rbm_list = []

# Size of inputs is the number of inputs in the training set
input_size = inpX.shape[1]

# For each RBM we want to generate
for i, size in enumerate(RBM_hidden_sizes):
    print 'RBM: ', i, ' ', input_size, '->', size
    rbm_list.append(RBM(input_size, size, epochs=20, learning_rate=1.0, batchsize=256))
    input_size = size

# For each RBM in our list
for rbm in rbm_list:
    print 'New RBM:'
    # Train a new one
    rbm.train(inpX)
    # Return the output layer
    inpX = rbm.rbm_outpt(inpX)
    print inpX.shape

# for rbm in rbm_list:
#     # print rbm.w.shape, rbm.vb.shape, rbm.hb.shape
#     # print type(rbm.w.shape)
#     np.savetxt('w.txt', rbm.w, delimiter=',')

nNet = NN(RBM_hidden_sizes, trX, trY)
nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
nNet.train()
