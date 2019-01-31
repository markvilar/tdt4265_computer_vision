import mnist
import numpy as np


def partition_dataset(train_data, test_data, train_size, test_size):
    '''
    Partitions the training and test sets into subsets.
    '''
    X_train, Y_train = train_data
    X_test, Y_test = test_data
    train_samps, train_feats = X_train.shape
    test_samps, test_feats = X_test.shape

    if train_size > train_samps or train_size < 0:
        train_size = train_samps
    if test_size > test_samps or test_size < 0:
        test_size = test_samps

    X_train = X_train[0:train_size]
    Y_train = Y_train[0:train_size]
    X_test = X_test[test_samps-test_size:]
    Y_test = Y_test[test_samps-test_size:]

    return (X_train, Y_train), (X_test, Y_test)


def create_validation_set(train_data, val_perc):
    '''
    Partitions the training set into a smaller training set and a validation
    set.
    '''
    X_train, Y_train = train_data
    n_samps, n_feats = X_train.shape
    if val_perc >= 1 or val_perc < 0:
        val_perc=0.1
    n_val = int(np.floor(val_perc * n_samps))
    val_idxs = np.random.permutation(np.arange(n_samps))[:n_val]
    X_val = X_train[val_idxs,:]
    Y_val = Y_train[val_idxs]
    X_train = np.delete(X_train, val_idxs, axis=0)
    Y_train = np.delete(Y_train, val_idxs, axis=0)
    return (X_train, Y_train), (X_val, Y_val)


def remove_digit(train_data, test_data, digit):
    '''
    Removes a specific digit from the dataset.
    '''
    X_train, Y_train = train_data
    X_test, Y_test = test_data
    b_train = (Y_train != digit)
    b_test = (Y_test != digit)

    X_train = X_train[b_train]
    Y_train = Y_train[b_train]
    X_test = X_test[b_test]
    Y_test = Y_test[b_test]

    return (X_train, Y_train), (X_test, Y_test)


def normalize_data(data):
    '''
    Normalizes the data so that each feature has a mean of 0 and a standard
    deviation of 1.
    '''
    X, Y = data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    return (X, Y)


def append_ones(data):
    '''
    Adds a 1 to the feature vector, so that the bias for each node is given
    by the weight of that input.
    '''
    X, Y = data
    n_samps, n_feats = X.shape
    ones = np.full((n_samps, 1), 1)
    X = np.append(X, ones, axis=1)
    return (X, Y)


def binary_class(data, true_class):
    '''
    Changes the targets to a binary encoding.
    '''
    X, Y = data
    for i in range(Y.shape[0]):
        Y[i] = np.int_(Y[i] == true_class)
    return (X, Y)


def process_data(task, train_size=20000, test_size=2000, val_perc=0.1):
    '''
    Creates the datasets to be used in the logistic regression task.
    '''
    if task == 'logistic_regression':
        excluded=[0,1,4,5,6,7,8,9]
        true_class=2
    X_train, Y_train, X_test, Y_test = mnist.load()
    train_data, test_data = (X_train, Y_train), (X_test, Y_test)
    train_data, test_data = partition_dataset(train_data, test_data, train_size,
                                              test_size)
    for digit in excluded:
        train_data, test_data = remove_digit(train_data, test_data, digit)

    train_data, val_data = create_validation_set(train_data, val_perc)
    train_data = normalize_data(train_data)
    val_data = normalize_data(val_data)
    test_data = normalize_data(test_data)
    train_data = append_ones(train_data)
    val_data = append_ones(val_data)
    test_data = append_ones(test_data)

    if task == 'logistic_regression':
        train_data = binary_class(train_data, true_class)
        val_data = binary_class(val_data, true_class)
        test_data = binary_class(test_data, true_class)
    return train_data, val_data, test_data
