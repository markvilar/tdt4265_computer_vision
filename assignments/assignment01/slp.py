import data_preprocess as dp
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


class SLP(object):
    def __init__(self, in_size, out_size, eta, T, batch_size):
        '''
        Initializes the network with normally distributed weights with a mean
        of 0 and variance of 1.
        Params:
            weights     - np.array  - weights between layers
            layers      - list      - list of layer sizes
            eta         - float     - learn rate
            batch_size  - int       - batch size
            activ_fun   - string    - activation function
            epochs      - int       - number of training epochs performed
        '''
        self.weights = np.random.normal(size=(in_size, out_size))
        self.layers = [in_size, out_size]
        self.eta = eta
        self.T = T
        self.batch_size = batch_size

    def shuffle(self, train_data):
        X_train, Y_train = train_data
        randomize = np.arange(len(X_train))
        np.random.shuffle(randomize)
        X_train = X_train[randomize]
        Y_train = Y_train[randomize]
        return (X_train, Y_train)

    def create_batches(self, train_data):
        ''' Divides the training data into batches of size batch_size.'''
        batches = []
        X_train, Y_train = train_data
        n_splits = len(X_train) // self.batch_size
        rem = len(X_train) % n_splits

        rem_batch = (X_train[-rem:], Y_train[-rem:])
        X_train = X_train[:-rem]
        Y_train = Y_train[:-rem]
        X_splits = np.split(X_train, n_splits)
        Y_splits = np.split(Y_train, n_splits)

        for X, Y in zip(X_splits, Y_splits):
            batch = (X, Y)
            batches.append(batch)
        batches.append(rem_batch)
        return batches

    def logistic_forward(self, x):
        ''' Feedforward method for logistic regression '''
        input = np.matmul(np.transpose(self.weights), x)
        return sigmoid(input)

    def softmax_forward(self, x):
        ''' Feedforward method for softmax regression '''
        input = np.matmul(np.transpose(self.weights), x)
        return softmax(input)

    def backwards(self, loss_grad):
        eta = self.eta / (1 + self.t/self.T)
        self.weights -= eta * loss_grad

    def logistic_SGD(self, epochs, train_data, val_data, test_data):
        ''' Stochastic gradient descent algorithm for binary logistic
        regression with early stopping. '''
        for epoch in range(epochs):
            train_data = self.shuffle(train_data)
            batches = self.create_batches(train_data)
            for batch in batches:
                loss_grad = 0
                n = len(batch[0])
                i in range(n):
                    x = batch[0][i]
                    y = batch[1][i]
                    output = logistic_forward(x)
                    loss_grad +=
            #loss = self.logistic_validate(val_data)
            #print('Epoch {}/{}... Loss {:.3}'.format(epoch, epochs, loss))

    def softmax_SGD(self):
        pass

    def logistic_evaluate(self):
        pass

    def softmax_evaluate(self):
        pass

    def visualize_weights(self, fill_value=0, stride=5):
        '''
        Visualizes the weights of the neural network as images.
        '''
        weights = self.weights
        (m,n) = weights.shape
        print(n)
        k = int(np.sqrt(m))
        if k != np.sqrt(m):
            k += 1
            j = k**2 - m
            fill = np.full((j,n), fill_value)
            weights = np.append(weights, fill, axis=0)
        fig, axes = plt.subplots(n, squeeze=False)
        for i in range(n):
            axes[i,0].imshow(np.reshape(weights[:,i], (k, k)))
        plt.show()

    def save_network(self):
        pass

    def load_network(self):
        pass


def main():
    in_size = 785
    out_size = 10
    epochs = 30
    eta = 0.01
    T = 10
    batch_size = 30
    task = 'logistic_regression'
    train_data, val_data, test_data = dp.process_data(task)
    network = SLP(in_size, out_size, eta, T, batch_size)
    train_data = network.shuffle(train_data)
    batches = network.create_batches(train_data)
    for batch in batches:
        print(len(batch[0]))
        print(len(batch[1]))
    print(batches[-1][0][0])
    print(batches[-1][1][0])


if __name__ == '__main__':
    main()
