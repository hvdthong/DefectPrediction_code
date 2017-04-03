import os
import numpy as np
import cPickle as pickle
import sys


def get_data(path):
    train, test = [], []
    with open(path + '/train.txt') as f:
        lines = f.read().splitlines()
        lines = [l.split(',') for l in lines]
        lines = [l[1:] for l in lines]
        train = lines

    with open(path + '/test.txt') as f:
        lines = f.read().splitlines()
        lines = [l.split(',') for l in lines]
        lines = [l[1:] for l in lines]
        test = lines
    return train, test


def dictionary(train, test):
    dict_ = []
    total = train + test
    words = [w for l in total for w in l]
    lengths = [len(l) for l in total]

    for w in words:
        if w not in dict_:
            dict_.append(w)
    return sorted(dict_), max(lengths)


def mapping(data, dict, max_len):
    ftrs = []
    for l in data:
        ftr = [0] * max_len
        for i in range(0, len(l)):
            index = dict.index(l[i])
            ftr[i] = index / (float(max_len))
        ftrs.append(np.array(ftr))
    return np.array(ftrs)


def save_variables(folders, path):
    for f in folders:
        train_, test_ = get_data(path + f)
        dict_, max_len_ = dictionary(train_, test_)

        ftr_train = mapping(train_, dict_, max_len_)
        ftr_test = mapping(test_, dict_, max_len_)

        pickle.dump(ftr_train, open(path + f + '/ftr_train.p', 'wb'))
        pickle.dump(ftr_test, open(path + f + '/ftr_test.p', 'wb'))

        print f, len(dict_), max_len_


def load_variables(folder):
    train = pickle.load(open(folder + '/ftr_train.p', 'rb'))
    test = pickle.load(open(folder + '/ftr_test.p', 'rb'))
    return train, test


if __name__ == '__main__':
    path_ = '../data/'
    folders_ = os.listdir(path_)
    # save_variables(folders_, path_)
    for f in folders_:
        train, test = load_variables(f + path_)
        print train.shape, test.shape

        print train[0]
        print test[0]
