#################################
# Your name: Michael Hasson
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper_hinge():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784',as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """

    n=np.shape(data)[0]
    w = np.zeros_like(data[0])
    for t in range(1,T+1):
        eta=eta_0/t
        i=np.random.choice(n)
        if labels[i]*np.dot(w,data[i])<1:
            w*=(1-eta)
            w+=eta*C*labels[i]*data[i]
        else:
            w*=(1-eta)
    return w

def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """

    n=np.shape(data)[0]
    w_class = [np.zeros_like(data[0]) for x in range(10)]
    for t in range(1, T + 1):
        eta = eta_0 / t
        i = np.random.choice(n)
        if labels[i] * np.dot(w, data[i]) < 1:
            w *= (1 - eta)
            w += eta * C * labels[i] * data[i]
        else:
            w *= (1 - eta)
    return w
    return

def accurate(w,x,y):
    c=-1
    if np.dot(w, x) > 0:
        c=1
    if c == y:
        return 1
    return 0

def compute_accuracy(data,labels,w):
    return np.average(np.array([accurate(w,x,y) for x, y in zip(data, labels)]))

def compute_emp_error(data,labels,w,C):
    return np.average(np.array([C * max(0, 1 - y * np.dot(w, x)) + 0.5 * np.linalg.norm(w) for x, y in zip(data, labels)]))

def find_best_eta (T, C, eta0_vals , train_data, train_labels, validation_data, validation_labels):
    accuracy_lst=np.array([])
    for eta in eta0_vals:
        accu=0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            accu+=compute_accuracy(validation_data, validation_labels, w)
        accu/=10
        accuracy_lst=np.append(accuracy_lst,accu)

    best_etha=eta0_vals[np.argmax(accuracy_lst)]
    print("Best eta=",best_etha," Accuracy=",np.max(accuracy_lst))
    plt.plot(eta0_vals,accuracy_lst)
    plt.xlabel("η0")
    plt.ylabel("avrg. accuracy of validation")
    plt.xscale("log")
    plt.ylim(0,1)
    plt.show()
    return best_etha


def find_best_C(T, eta,C_vals, train_data, train_labels, validation_data, validation_labels):
    accuracy_lst=np.array([])
    for C in C_vals:
        accu=0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            accu+=compute_accuracy(validation_data, validation_labels, w)
        accu/=10
        accuracy_lst=np.append(accuracy_lst,accu)

    best_C=C_vals[np.argmax(accuracy_lst)]
    print("Best C=",best_C," Accuracy=",np.max(accuracy_lst))
    plt.plot(eta0_vals,accuracy_lst)
    plt.xlabel("C")
    plt.ylabel("avrg. accuracy of validation")
    plt.xscale("log")
    plt.ylim(0.97,0.990)
    plt.show()
    return best_C


if __name__=="__main__":
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    T=1000
    C = 1.0


    eta0_vals = [10**i for i in range(-7,8)]
    #best_eta =
    find_best_eta (T, C, eta0_vals , train_data, train_labels, validation_data, validation_labels)
    best_eta=1
    C_vals = eta0_vals
    best_C = find_best_C(T, best_eta,C_vals, train_data, train_labels, validation_data, validation_labels)


    #T=20000
    #w = SGD_hinge(train_data, train_labels, best_C, best_eta, T)
    #plt.imshow(w.reshape(28,28))
    #plt.show()

    #
    #print("accuracy of the best classifier on the test set:",compute_accuracy(test_data,test_labels,w))
