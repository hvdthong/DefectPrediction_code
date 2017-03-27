import tensorflow as tf
import numpy as np


class autoencoder(object):
    def __init__(self, input_size, hidden_size, epochs, learning_rate, batch_size):
        # Defining the hyperparameters
        self.inputfeature = input_size  # Size of input
        self.hidden = hidden_size  # Size of output
        self.epochs = epochs  # Amount of training iterations
        self.learning_rate = learning_rate  # The step used in gradient descent
        self.batch_size = batch_size  # The size of how much data will be used for training per sub iteration
        self.display_step = 1

        self.weights = []
        self.biases = []

    def initialPara(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.inputfeature, self.hidden])),
            'decoder_h1': tf.Variable(tf.random_normal([self.hidden, self.inputfeature]))
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.hidden])),
            'decoder_b1': tf.Variable(tf.random_normal([self.inputfeature]))
        }
        self.weights = weights
        self.biases = biases

    def encoder(self, x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        return layer

    def decoder(self, x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        return layer

    def train(self, X):

        self.X_ = tf.placeholder('float', [None, X.shape[1]])
        batch_size = self.batch_size

        self.initialPara()
        # tf_weights_ = tf.Variable(self.weights)
        # tf_biases_ = tf.Variable(self.biases)

        encoder_op = self.encoder(self.X_)
        decoder_op = self.decoder(encoder_op)

        y_pred = decoder_op
        y_true = self.X_

        # define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)

        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)
        # launch the graph

        total_batch = int(X.shape[0] / batch_size)

        weights_, biases_, costs = [], [], []

        # training cycle
        for epoch in range(self.epochs):
            # loop over all batches
            for i in range(total_batch):
                batch_xs = X[i * batch_size:(i + 1) * batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={self.X_: batch_xs})

                # encoders.append(sess.run([encoder_op], feed_dict={self.X_: batch_xs}))
                # decoders.append(sess.run([decoder_op], feed_dict={self.X_: batch_xs}))

                # weights_.append(sess.run([self.weights], feed_dict={self.X_:batch_xs}))
                costs.append(c)
            w_encode = self.weights['encoder_h1'].eval()
            w_decode = self.weights['decoder_h1'].eval()
            weights_.append({'encoder_h1': w_encode, 'decoder_h1': w_decode})
            # display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        print("optimization finished!!")

        self.encoderOp = encoder_op
        self.decoderOp = decoder_op
        self.weights_out = np.array(weights_)
        self.costs = costs
