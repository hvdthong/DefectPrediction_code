import tensorflow as tf
import numpy as np


class autoencoder(object):
    def __init__(self, input_size, hidden_size, output_size, epochs, learning_rate, batch_size):
        # Defining the hyperparameters
        self.input = input_size  # Size of input
        self.hidden = hidden_size  # Size of dimension reduction
        self.output = output_size  # Size of the class label
        self.epochs = epochs  # Amount of training iterations
        self.learning_rate = learning_rate  # The step used in gradient descent
        self.batch_size = batch_size  # The size of how much data will be used for training per sub iteration
        self.display_step = 1

        self.weights = []
        self.biases = []

    def initialPara(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.input, self.hidden])),
            'decoder_h1': tf.Variable(tf.random_normal([self.hidden, self.input]))
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.hidden])),
            'decoder_b1': tf.Variable(tf.random_normal([self.input]))
        }
        self.weights = weights
        self.biases = biases

    def encoder(self, x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        return layer

    def decoder(self, x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        return layer

    def train(self, X, Y, X_test, Y_test):
        self.X_ = tf.placeholder('float', [None, X.shape[1]])
        self.Y_ = tf.placeholder('float', [None, Y.shape[1]])
        batch_size = self.batch_size

        self.initialPara()
        # tf_weights_ = tf.Variable(self.weights)
        # tf_biases_ = tf.Variable(self.biases)

        encoder_op = self.encoder(self.X_)
        decoder_op = self.decoder(encoder_op)

        # unsupervised learning
        y_pred_un = decoder_op
        y_true_un = self.X_

        # supervised learning
        W = tf.Variable(tf.random_normal([X.shape[1], Y.shape[1]]))
        b = tf.Variable(tf.random_normal([Y.shape[1]]))
        y_pred = tf.nn.softmax(tf.matmul(self.X_, W) + b)  # Softmax
        y_true = self.Y_

        # define loss and optimizer, minimize the squared error
        cost_un = tf.reduce_mean(tf.pow(y_pred_un - y_true_un, 2))
        cost_su = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
        total_cost = cost_un + cost_su

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(total_cost)

        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)
        # launch the graph

        total_batch = int(X.shape[0] / batch_size)
        cost_uns, cost_sus = [], []

        weights_, biases_, costs = [], [], []

        # training cycle
        for epoch in range(self.epochs):
            # loop over all batches
            for i in range(total_batch):
                batch_xs = X[i * batch_size:(i + 1) * batch_size]
                batch_ys = Y[i * batch_size:(i + 1) * batch_size]
                _, c, u, s = sess.run([optimizer, total_cost, cost_un, cost_su],
                                      feed_dict={self.X_: batch_xs, self.Y_: batch_ys})

                cost_uns.append(u)
                cost_sus.append(s)

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

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({self.X_: X_test, self.Y_: Y_test}))

        self.encoderOp = encoder_op
        self.decoderOp = decoder_op
        self.weights_out = np.array(weights_)
        self.costs = costs
        self.costs_un = cost_uns
        self.costs_su = cost_sus

    def output(self, X):
        input_ = tf.constant(X)
        encoder_ = self.encoder(input_)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            return sess.run(encoder_)


class autoencoder_advance(object):
    def __init__(self, input_size, hiddens, output_size, epochs, learning_rate, batch_size):
        # Defining the hyperparameters
        self.input = input_size  # Size of input
        self.hiddens = hiddens  # Hidden layers & number of nodes in hidden
        self.output = output_size  # Size of the class label
        self.epochs = epochs  # Amount of training iterations
        self.learning_rate = learning_rate  # The step used in gradient descent
        self.batch_size = batch_size  # The size of how much data will be used for training per sub iteration
        self.display_step = 1

        self.weights = []
        self.biases = []

    def initialPara(self):
        weights_en, biases_en = [], []
        weights_de, biases_de = [], []
        for i in range(0, len(self.hiddens) - 1):
            if i == 0:
                layer = tf.Variable(tf.random_normal([self.input, self.hiddens[i]]))
            else:
                layer = tf.Variable(tf.random_normal([self.hiddens[i], self.hiddens[i + 1]]))
            bias = tf.Variable(tf.random_normal([self.hiddens[i]]))
