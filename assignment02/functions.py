import numpy as np


def sigmoid(x):
    ''' The sigmoid activation function. '''
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    ''' Derivative of the sigmoid activation function. '''
    sigma = sigmoid(x)
    return np.multiply(sigma, (1 - sigma))

def ReLU(x):
    ''' The ReLU activation function. '''
    return np.where(x > 0, x, 0)

def ReLU_prime(x):
    ''' Derivative of the ReLU activation function. '''
    ones = np.ones(x.shape)
    return np.where(x > 0, ones, 0)

def ELU(x):
    ''' The ELU activation function. '''
    alpha = 1
    a = np.zeros(shape=x.shape)
    for i in range(len(x)):
        if x[i] > 0:
            a[i] = x[i]
        else:
            a[i] = alpha * (np.exp(x[i])-1)
    return a

def ELU_prime(x):
    ''' Derivative of the ELU activation function. '''
    alpha = 1
    a = np.zeros(shape=x.shape)
    for i in range(len(x)):
        if x[i] > 0:
            a[i] = 1
        else:
            a[i] = alpha * np.exp(x[i])
    return a

def tanh(x):
    ''' The tanh activation function. '''
    return np.tanh(x)

def tanh_prime(x):
    ''' Derivative of the tanh activation function. '''
    return 1 - np.power(tanh(x), 2)

def softmax(x):
    ''' The softmax activation function, numerically stable implementation '''
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def softmax_prime(x):
    ''' Derivative of the softmax activation function. '''
    y = softmax(x)
    return np.multiply(y, (1 - y))

def cross_entropy(x, y):
    ''' The cross entropy loss function. '''
    return np.mean(np.nan_to_num(-y*np.log(x)))

def cross_entropy_prime(x, y):
    ''' The derivative of the cross entropy function with respect to the
    input of the output layer. '''
    return -(y-x)
