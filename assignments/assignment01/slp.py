import data_preprocess as dp
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    ''' The sigmoid function. '''
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    ''' Derivative of the sigmoid function. '''
    sigma = sigmoid(x)
    return sigma*(1-sigma)


def softmax(x):
    ''' The softmax function. '''
    exps = np.exp(x)
    return exps/np.sum(exps)


def cross_entropy(y, output):
    # Fixes division by zero bug
    if output == 1:
        output = 0.9999999
    elif output == 0:
        output = 0.0000001
    return -(y*np.log(output) + (1-y)*np.log(1-output))


def vector_length(v):
    squared_sum = 0
    for i in range(len(v)):
        squared_sum = v[i]**2
    return squared_sum

class SLP(object):
    def __init__(self, in_size, out_size, eta, T, batch_size, gamma=0):
        '''
        Initializes the network with normally distributed weights with a mean
        of 0 and variance of 1.
        Params:
            weights     - np.array  - weights between layers
            layers      - list      - list of layer sizes
            eta         - float     - learn rate
            T           - int       - annealing learn rate parameter
            batch_size  - int       - batch size
            gamma       - float     - L2 regularization parameter
        '''
        self.weights = np.random.normal(size=(in_size, out_size))
        self.layers = [in_size, out_size]
        self.eta = eta
        self.T = T
        self.batch_size = batch_size
        self.gamma = gamma

    def shuffle(self, train_data):
        ''' Shuffles the order of the training data. '''
        X_train, Y_train = train_data
        randomize = np.arange(len(X_train))
        np.random.shuffle(randomize)
        X_train = X_train[randomize]
        Y_train = Y_train[randomize]
        return (X_train, Y_train)

    def create_batches(self, train_data):
        ''' Divides the training data into batches.'''
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

    def backwards(self, epoch, loss_grad):
        ''' Updates the weights of the perceptron. '''
        eta = self.eta / (1 + epoch/self.T)
        self.weights = self.weights - eta * (loss_grad + 2*self.gamma*self.weights)

    def logistic_forward(self, x):
        ''' Feedforward method for logistic regression. '''
        return sigmoid(np.dot(x, self.weights))

    def logistic_validate(self, val_data):
        loss = 0
        correct = 0
        n = len(val_data[0])
        for i in range(n):
            x = val_data[0][i]
            y = val_data[1][i]
            output = self.logistic_forward(x)
            output = float(output)
            y = int(y)
            loss += (1/n) * cross_entropy(y, output)
            if abs(output-y) < 0.5:
                correct += 1
        perc = correct / n
        return loss, perc, self.weights

    def logistic_evaluate(self, test_data):
        loss = 0
        correct = 0
        n = len(test_data[0])
        for i in range(n):
            x = test_data[0][i]
            y = test_data[1][i]
            output = self.logistic_forward(x)
            output = float(output)
            y = int(y)
            loss += (1/n) * cross_entropy(y, output)
            if abs(output-y) < 0.5:
                correct += 1
        perc = correct / n
        return loss, perc

    def logistic_SGD(self, epochs, train_data, val_data, test_data):
        ''' Stochastic gradient descent algorithm for binary logistic
        regression with early stopping. '''
        log = []
        for epoch in range(epochs):
            train_data = self.shuffle(train_data)
            batches = self.create_batches(train_data)
            for batch in batches:
                loss_grad = 0
                n = len(batch[0])
                for i in range(n):
                    x = batch[0][i]
                    y = np.array(batch[1][i])
                    output = self.logistic_forward(x)
                    error = output - y
                    error = np.atleast_2d(error)
                    x = np.atleast_2d(x).transpose()
                    loss_grad += np.dot(np.atleast_2d(x), error)
                self.backwards(epoch, loss_grad)
                loss_grad = 0
            val_loss, val_perc, weights = self.logistic_validate(val_data)
            train_loss, train_perc = self.logistic_evaluate(train_data)
            test_loss, test_perc = self.logistic_evaluate(test_data)
            w_length = vector_length(self.weights)
            print('Epoch:{:2}/{} Train.loss: {:4.3} Val.loss: {:4.3} Test.loss: {:4.3}'\
            .format(epoch+1, epochs, train_loss, val_loss, test_loss))
            log.append([epoch, train_loss, val_loss, test_loss, \
                        train_perc, val_perc, test_perc, w_length])
        return log

    def softmax_forward(self, x):
        ''' Feedforward method for softmax regression '''
        input = np.matmul(np.transpose(self.weights), x)
        return softmax(input)

    def softmax_validate(self, x):
        pass

    def softmax_evaluate(self):
        pass

    def softmax_SGD(self):
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
        fig, axes = plt.subplots(n, squeeze=False)
        for i in range(n):
            axes[i,0].imshow(np.reshape(weights[:,i], (k, k)))
        plt.show()


def main():
    task = '2.2'
    if task == '2.1':
        in_size = 785
        out_size = 1
        epochs = 50
        eta = 0.001
        gamma = 0.01
        T = 30
        batch_size = 30
        train_data, val_data, test_data = dp.process_data('logistic_regression')
        network = SLP(in_size, out_size, eta, T, batch_size, gamma)
        log = network.logistic_SGD(epochs, train_data, val_data, test_data)
        epochs = []
        train_loss = []
        val_loss = []
        test_loss = []
        train_perc = []
        val_perc = []
        test_perc = []
        for i in range(len(log)):
            epochs.append(log[i][0])
            train_loss.append(log[i][1])
            val_loss.append(log[i][2])
            test_loss.append(log[i][3])
            train_perc.append(log[i][4])
            val_perc.append(log[i][5])
            test_perc.append(log[i][6])
        plt.plot(epochs, train_loss, epochs, val_loss, epochs, test_loss)
        plt.xlabel('Training epochs')
        plt.ylabel('Cross entropy loss')
        plt.title('Task 2.1 a), eta = 0.001, T = 30, batch_size = 30')
        plt.legend(['Training loss', 'Validation loss', 'Test loss'])
        plt.show()
        plt.plot(epochs, train_perc, epochs, val_perc, epochs, test_perc)
        plt.xlabel('Training epochs')
        plt.ylabel('Percentage, correctly classified')
        plt.title('Task 2.1, eta = 0.001, T = 30, batch_size = 30')
        plt.legend(['Training set', 'Validation set', 'Test set'])
        plt.show()
    elif task == '2.2':
        in_size = 785
        out_size = 1
        epochs = 30
        eta = 0.001
        gamma = 0.01
        T = 30
        batch_size = 30
        train_data, val_data, test_data = dp.process_data('logistic_regression')
        network1 = SLP(in_size, out_size, eta, T, batch_size, gamma=0.01)
        network2 = SLP(in_size, out_size, eta, T, batch_size, gamma=0.001)
        network3 = SLP(in_size, out_size, eta, T, batch_size, gamma=0.0001)
        log1 = network1.logistic_SGD(epochs, train_data, val_data, test_data)
        log2 = network2.logistic_SGD(epochs, train_data, val_data, test_data)
        log3 = network3.logistic_SGD(epochs, train_data, val_data, test_data)
        epochs = []
        val_percs1 = []
        val_percs2 = []
        val_percs3 = []
        weight_lengths1 = []
        weight_lengths2 = []
        weight_lengths3 = []
        for i in range(len(log1)):
            epochs.append(log1[i][0])
            val_percs1.append(log1[i][5])
            val_percs2.append(log2[i][5])
            val_percs3.append(log3[i][5])
            weight_lengths1.append(log1[i][7])
            weight_lengths2.append(log2[i][7])
            weight_lengths3.append(log3[i][7])
        plt.plot(epochs, val_percs1, epochs, val_percs2, epochs, val_percs3)
        plt.xlabel('Training epochs')
        plt.ylabel('Percentage, correctly classified')
        plt.title('Task 2.2b, eta = 0.001, T = 30, batch_size = 30')
        plt.legend(['gamma = 0.01', 'gamma = 0.001', 'gamma = 0.0001'])
        plt.show()
        plt.plot(epochs, weight_lengths1, epochs, weight_lengths2, epochs, weight_lengths3)
        plt.xlabel('Training epochs')
        plt.ylabel('Weight length')
        plt.title('Task 2.2c, eta = 0.001, T = 30, batch_size = 30')
        plt.legend(['gamma = 0.01', 'gamma = 0.001', 'gamma = 0.0001'])
        plt.show()

        network1.visualize_weights()
        network2.visualize_weights()
        network3.visualize_weights()


if __name__ == '__main__':
    main()
