import numpy as np
import tensorflow as tf

# print np.sign([-5., 4.5])

# p = [1, 4, 6, 3, 8]
# a = np.random.uniform(p)
#
# k = np.random.uniform(-1,0,1000)
# print a
# print k

rand = tf.random_uniform([500])
value = tf.nn.relu(rand)
sess = tf.Session()
init = tf.initialize_all_variables()
# print sess.run(rand)
with sess.as_default():
    print len(rand.eval())
    print rand.eval()
    print value.eval()

