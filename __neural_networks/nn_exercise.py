from scipy.io import loadmat
from scipy import optimize as opt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('bmh')

data = loadmat('data/ex4data1.mat')
theta = loadmat('data/ex4weights.mat')

X = data['X']
y = data['y']
theta1 = theta['Theta1']
theta2 = theta['Theta2']

y_n = pd.get_dummies(y.flatten())
nn_params = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))


def feedforward(theta1, theta2, m, X):
    bias = np.ones((m, 1))
    a1 = np.hstack((bias, X))
    a2 = sigmoid(a1.dot(theta1.T))
    a2 = np.hstack((bias, a2))
    a3 = sigmoid(a2.dot(theta2.T))
    h = a3
    return h


def cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, λ=1):

    m = len(y)
    _slice = hidden_layer_size * (input_layer_size + 1)

    th1 = np.reshape(
        nn_params[: _slice], (hidden_layer_size, input_layer_size + 1), order='F')

    th2 = np.reshape(
        nn_params[_slice:], (num_labels,        hidden_layer_size + 1), order='F')

    h = feedforward(th1, th2, m, X)

    # cost

    cost = np.sum((np.multiply(y, np.log(h))) +
                  (np.multiply(1-y, np.log(1-h))))

    # regularization
    L2_on_TH1 = np.power(th1[:, 1:], 2)
    L2_on_TH2 = np.power(th2[:, 1:], 2)

    reg_sum_1 = np.sum(np.sum(L2_on_TH1, axis=1))
    reg_sum_2 = np.sum(np.sum(L2_on_TH2, axis=1))

    return np.sum(cost / (-m)) + (reg_sum_1 + reg_sum_2) * lmbda / (2*m)


input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lmbda = 1
