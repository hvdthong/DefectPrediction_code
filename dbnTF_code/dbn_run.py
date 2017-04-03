# Getting the MNIST data provided by Tensorflow
from tensorflow.examples.tutorials.mnist import input_data
from NN import NN
from rbm import RBM
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../services/')
from preprocessing import load_variables



def running_MNIST():
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

    for rbm in rbm_list:
        print rbm.w.shape, rbm.vb.shape, rbm.hb.shape
        print rbm.errors

        plt.plot(rbm.errors[0::100])
        plt.xlabel('Batch Number')
        plt.ylabel('Error')
        plt.legend(['Error'], loc='upper right')
        # plt.show()
        plt.savefig('Error.jpg')

        #     np.savetxt('w.txt', rbm.w, delimiter=',')

        # nNet = NN(RBM_hidden_sizes, trX, trY)
        # nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
        # nNet.train()


def running_data(folder):
    train, test = load_variables(folder)
    # print train.shape, test.shape

    RBM_hidden_sizes = [100]  # create 1 layers of RBM with size 100

    # Since we are training, set input as training data
    inpX = train

    # Create list to hold our RBMs
    rbm_list = []

    # Size of inputs is the number of inputs in the training set
    input_size = inpX.shape[1]
    batchsize = inpX.shape[0] / 100

    # For each RBM we want to generate
    for i, size in enumerate(RBM_hidden_sizes):
        print 'RBM: ', i, ' ', input_size, '->', size
        rbm_list.append(RBM(input_size, size, epochs=20, learning_rate=1.0, batchsize=batchsize))
        input_size = size

    # For each RBM in our list
    for rbm in rbm_list:
        print 'New RBM:'
        # Train a new one
        rbm.train(inpX)
        # Return the output layer
        inpX = rbm.rbm_outpt(inpX)
        print inpX.shape

    for rbm in rbm_list:
        print rbm.w.shape, rbm.vb.shape, rbm.hb.shape
        pickle.dump(rbm.w, open(folder + '/w.p', 'wb'))
        pickle.dump(rbm.vb, open(folder + '/vb.p', 'wb'))

        print rbm.errors

        plt.plot(rbm.errors[0::100])
        plt.xlabel('Batch Number')
        plt.ylabel('Error')
        plt.legend(['Error'], loc='upper right')
        # plt.show()
        plt.savefig('Error_%s_layers_%d_sizes_%d.jpg'
                    % (folder.split('/')[-1], len(RBM_hidden_sizes), RBM_hidden_sizes[0]))


def output_data(folder):
    ftr_train, ftr_test = load_variables(folder)
    w, b = pickle.load(open(folder + '/w.p', 'rb')), pickle.load(open(folder + '/vb.p', 'rb'))
    print w.shape, b.shape


if __name__ == '__main__':
    path_ = '../data/'
    folders_ = os.listdir(path_)
    # for f in folders_:
    #     running_data(path_ + f)

    for f in folders_:
        output_data(path_ + f)

        # running_MNIST()
