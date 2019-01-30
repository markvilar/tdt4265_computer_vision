import mnist
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    '''
    The sigmoid function.
    '''
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    '''
    Derivative of the sigmoid function.
    '''
    sigma = sigmoid(x)
    return sigma*(1-sigma)


def softmax(x):
    '''
    The softmax function.
    '''
    exps = np.exp(x)
    return exps/np.sum(exps)


def partition_dataset(X_train, Y_train, X_test, Y_test, train_size, test_size):
    '''
    Partitions the training and test sets into subsets.
    '''
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

    return X_train, Y_train, X_test, Y_test


def create_validation_set(X_train, Y_train, val_perc):
    '''
    Partitions the training set into a smaller training set and a validation
    set.
    '''
    n_samps, n_feats = X_train.shape
    if val_perc >= 1 or val_perc < 0:
        val_perc=0.1
    n_val = int(np.floor(val_perc * n_samps))
    val_idxs = np.random.permutation(np.arange(n_samps))[:n_val]
    X_val = X_train[val_idxs,:]
    Y_val = Y_train[val_idxs]
    X_train = np.delete(X_train, val_idxs, axis=0)
    Y_train = np.delete(Y_train, val_idxs, axis=0)
    return X_train, Y_train, X_val, Y_val


def remove_digit(X_train, Y_train, X_test, Y_test, digit):
    '''
    Removes a specific digit from the dataset.
    '''
    b_train = Y_train != digit
    b_test = Y_test != digit

    X_train = X_train[b_train]
    Y_train = Y_train[b_train]
    X_test = X_test[b_test]
    Y_test = Y_test[b_test]

    return X_train, Y_train, X_test, Y_test


def normalize_data(x):
    '''
    Normalizes the data so that each feature has a mean of 0 and a standard
    deviation of 1.
    '''
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1
    x = (x - mean) / std
    return x


def append_ones(x):
    '''
    Adds a 1 to the feature vector, so that the bias for each node is given
    by the weight of that input.
    '''
    rows, cols = x.shape
    ones = np.full((rows, 1), 1)
    return np.append(x, ones, axis=1)


def binary_classification_targets(y, true_class):
    '''
    Changes the targets to a binary encoding.
    '''
    for i in range(y.shape[0]):
        y[i] = np.int_(y[i] == true_class)
    return y


def logistic_regression_data_preprocess(X_train, Y_train, X_test, Y_test,
                                        train_size=20000, test_size=2000,
                                        val_perc=0.1,
                                        excluded=[0,1,4,5,6,7,8,9],
                                        true_class=2):
    '''
    Creates the datasets to be used in the logistic regression task.
    '''
    X_train, Y_train, X_test, Y_test = partition_dataset(X_train, Y_train,
                                                         X_test, Y_test,
                                                         train_size, test_size)

    for i in excluded:
        X_train, Y_train, X_test, Y_test = remove_digit(X_train, Y_train,
                                                        X_test, Y_test, i)

    X_train, Y_train, X_val, Y_val = create_validation_set(X_train, Y_train,
                                                           val_perc)
    X_train = normalize_data(X_train)
    X_val = normalize_data(X_val)
    X_test = normalize_data(X_test)
    X_train = append_ones(X_train)
    X_val = append_ones(X_val)
    X_test = append_ones(X_test)
    Y_train = binary_classification_targets(Y_train, true_class)
    Y_val = binary_classification_targets(Y_val, true_class)
    Y_test = binary_classification_targets(Y_test, true_class)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def logistic_regression():
    pass


def softmax_regression():
    pass


class Optimizer(object):
    def __init__(self):
        pass


class SLP(object):
    def __init__(self, in_size, out_size, learn_rate, T, batch_size,
                 activ_fun='sigmoid', optim='annealing'):
        '''
        Initializes the network with normally distributed weights with a mean
        of 0 and variance of 1.
        Params:
            weights     - np.array  - weights between layers
            layers      - list      - list of layer sizes
            learn_rate  - float     - learn rate
            batch_size  - int       - batch size
            activ_fun   - string    - activation function
            epochs      - int       - number of training epochs performed
            optim       - string    - optimizer algorithm
        '''
        self.weights = np.random.normal(size=(in_size, out_size))
        self.layers = [in_size, out_size]
        self.learn_rate = learn_rate
        self.T = T
        self.batch_size = batch_size
        self.activ_fun = activ_fun
        self.epochs = 0
        if optim == 'annealing' or optim == 'nesterov':
            self.optim = optim


    def forward(self, x):
        input = np.matmul(np.transpose(self.weights), x)
        if self.activ_fun == 'sigmoid':
            return sigmoid(input)
        elif self.activ_fun == 'softmax':
            return softmax(input)

    def backprop(self, ):
        if self.optim == 'annealing':
            pass
        elif self.optim == 'nesterov':
            pass

    def logistic_regression(self):
        pass

    def softmax_regression(self):
        pass

    def sgd(self):
        pass

    def minibatch_shuffle(self):
        pass

    def evaluate(self):
        pass

    def visualize_weights(self, fill_value=0, stride=5):
        '''
        Visualizes the weights of the neural network as images.
        '''
        weights = self.weights
        (m,n) = weights.shape
        k = int(np.sqrt(m))
        if k != np.sqrt(m):
            k += 1
            j = k**2 - m
            fill = np.full((j,n), fill_value)
            weights = np.append(weights, fill, axis=0)
        fig, axes = plt.subplots(1, n)
        for i in range(n):
            axes[i].imshow(np.reshape(weights[:,i], (k, k)))
        plt.show()

    def save_network(self):
        pass

    def load_network(self):
        pass


def main():
    network = SLP(785, 1, 0.001, 100, 20, 'sigmoid')
    X_train, Y_train, X_test, Y_test = mnist.load()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = logistic_regression_data_preprocess(X_train, Y_train, X_test, Y_test)
    for i in range(X_train.shape[0]):
        output = network.forward(X_train[i])


if __name__ == '__main__':
    main()
