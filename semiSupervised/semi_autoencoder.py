import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from autoencode import autoencoder

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

learning_rate = 0.01
batch_size = 256
training_epochs = 20
display_step = 1
hidden_unit = 128

# print trY.shape
# exit()

ae = autoencoder(trX.shape[1], trY.shape[1], hidden_unit, learning_rate=learning_rate
                 , epochs=training_epochs, batch_size=batch_size)
ae.train(trX, trY, teX, teY)
# print trY.shape
# print trY[0]

# print ae.encoderOp.shape
# print ae.decoderOp.shape
# k = ae.weights_out
# print ae.weights_out.shape

# print ae.costs

plt.plot(ae.costs[0::100])
plt.plot(ae.costs_su[0::100])
plt.plot(ae.costs_un[0::100])
plt.xlabel('Batch Number')
plt.ylabel('Error')
plt.legend(['Total Cost', 'Unsupervised Cost', 'Supervised Cost'], loc='upper right')
# plt.show()
plt.savefig('CostIteration.jpg')

# plt.plot(ae.costs_su)
# plt.xlabel('Batch Number')
# plt.ylabel('Error')
# plt.show()
#
# plt.plot(ae.costs_un)
# plt.xlabel('Batch Number')
# plt.ylabel('Error')
# plt.show()

# print ae.output(trX).shape
# print len(ae.output(trX)[0])
