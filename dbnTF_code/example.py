import numpy as np
# from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import Normalizer

a = [[1, 2, 4], [2, 5, 7]]
# print Normalizer(np.array(a), norm='l2')
print MinMaxScaler(np.array(a))

X = np.array(a)
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (X.max - min) + min
print X_std

# w = np.loadtxt('w.txt', delimiter=',')
# print w.shape
# print type(w)
