import numpy as np
import matplotlib.pyplot as plt
import functions as fun
import data_preprocess as data


def clear_grads(network):
    '''
    Removes all the calculated weight changes.
    '''
    for i in range(len(network.grads)):
        network.grads[i] = np.zeros(shape=network.grads[i].shape)

def numerical_grad(network, layer, input, label, epsilon=10**-2):
    '''
    Calculates the numerical approximation of the loss gradient wrt. weight
    ji of layer l. Returns the numerical approximation and the calculated one.
    Params:
        network - MLP, the neural network
        layer - int, the layer index
        input - np.array, input vector
        label - np.array, label vector
        epsilon - float, a small perturbation of the weight
    '''
    # Calculated gradient
    clear_grads(network)
    output = network.forward(input)
    loss, loss_grad = network.calc_loss(output, label)
    network.backward(loss_grad, 0, 1)
    dC_dw = network.grads[layer]
    clear_grads(network)

    dC_dw_approx = np.zeros(shape=dC_dw.shape)

    for k in range(network.weights[layer].shape[0]):
        for j in range(network.weights[layer].shape[1]):
            # Positively perturbate weight
            network.weights[layer][k][j] += epsilon
            output = network.forward(input)
            C1, loss_grad = network.calc_loss(output, label)
            # Negatively perturbate weight
            network.weights[layer][k][j] -= 2*epsilon
            output = network.forward(input)
            C2, loss_grad = network.calc_loss(output, label)
            # Restore weight
            dC_dw_approx[k][j] = (C1 - C2) / (2*epsilon)
            network.weights[layer][k][j] += epsilon

    max_difference = np.abs(dC_dw - dC_dw_approx).max()
    return max_difference


class MLP(object):
    def __init__(self, layers, act_funs, loss_fun, alpha, mu, T, idx_to_class, class_to_idx, xavier=True, p=0, gamma=0):
        '''
        Multi-layer perceptron class.
        Params:
            layers - list of ints, the size of the layers (not counting bias)
            act_funs - list of strings, the activation functions
            loss_fun - string, the loss function of the network
            alpha - float, the learning rate
            mu - float, momentum
            T - int, annealing learning rate time constant
            idx_to_class - dict {int, string}, index to class mapping
            class_to_idx - dict {string, int}, class to index mapping
            p - float, dropout probability
            gamma - float, L2 regularization weight decay parameter
        '''
        self.valid_act_funs = ['sigmoid', 'ReLU', 'ELU', 'tanh', 'softmax']
        self.valid_loss_funs = ['cross_entropy']
        self.check_valid(layers, act_funs, loss_fun, alpha, mu, T, p, gamma)
        self.act_funs = act_funs
        self.loss_fun = loss_fun
        self.weights = []
        self.grads = []
        self.del_weights = []
        self.prev_del_weights = []
        self.a = (len(layers)-1)*[None]
        self.z = (len(layers)-1)*[None]
        self.init_weights(layers, xavier)
        self.alpha = alpha
        self.mu = mu
        self.T = T
        self.idx_to_class = idx_to_class
        self.class_to_idx = class_to_idx
        self.p = p
        self.gamma = gamma
        self.log = {
                        'train_loss': [],
                        'val_loss': [],
                        'test_loss': [],
                        'train_acc': [],
                        'val_acc': [],
                        'test_acc': [],
                        'weights': []
                    }

    def check_valid(self, layers, act_funs, loss_fun, alpha, mu, T, p, gamma):
        '''
        Checks if the architecture of the network, i.e. the configuration
        of layers and activation functions, are valid. Also checks if
        hyperparameters have valid values.
        '''
        if (len(layers)-1) != len(act_funs):
            raise ValueError('The number of layers and activation functions \
                        are not equal.')
        if loss_fun not in self.valid_loss_funs:
            error_msg = 'Invalid loss function: ' + loss_fun
            raise ValueError(error_msg)
        for act_fun in act_funs:
            if act_fun not in self.valid_act_funs:
                error_msg = 'Invalid activation function: ' + act_fun
                raise ValueError(error_msg)
        if mu < 0:
            error_msg = 'Invalid momentum: ' + str(mu)
            raise ValueError(error_msg)
        if T <= 0:
            error_msg = 'Invalid annealing learn rate time constant: ' + str(T)
            raise ValueError(error_msg)
        if p < 0 or p > 1:
            error_msg = 'Invalid dropout probability: ' + str(p)
            raise ValueError(error_msg)
        if gamma < 0:
            error_msg = 'Invalid weight decay parameter: ' + str(gamma)
            raise ValueError(error_msg)

    def init_weights(self, layers, xavier):
        '''
        Initializes the weights and allocates the arrays for the loss gradient
        and weight changes.
        Params:
            layers - list of ints, the sizes of the layers
            xavier - boolean, triggers Xavier initialization
        '''
        for n_in, n_out in zip(layers[0:-1], layers[1:]):
            mean = 0
            if xavier:
                std = 1 / np.sqrt(n_in+1) # Add 1 to input size due to bias
            else:
                std = 1
            weights = np.random.normal(mean, std, size=(n_out, n_in+1))
            alloc = np.zeros(shape=weights.shape)
            self.weights.append(weights)
            self.grads.append(alloc)
            self.del_weights.append(alloc)
            self.prev_del_weights.append(alloc)

    def shuffle_data(self, train_data):
        '''
        Shuffles the order of the training data.
        Params:
            train_data - tuple of np.arrays, the training set
        '''
        X_train, Y_train = train_data
        randomize = np.arange(len(X_train))
        np.random.shuffle(randomize)
        X_train = X_train[randomize]
        Y_train = Y_train[randomize]
        return (X_train, Y_train)

    def create_mini_batches(self, train_data, batch_size):
        '''
        Divides the training data into mini batches.
        Params:
            train_data - tuple of np.arrays, the training set
            batch_size - int, the size of the mini batches
        '''
        batches = []
        X_train, Y_train = train_data
        n_splits = len(X_train) // batch_size
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

    def add_bias(self, x):
        '''
        Appends a 1 to x as a trick to add bias.
        Params:
            x - np.array, vector of neuron activations
        '''
        return np.append(x, 1)

    def forward(self, x):
        '''
        Performs a forward pass through the network.
        Params:
            x - np.array, the input vector
        '''
        for i in range(len(self.weights)):
            x = self.add_bias(x)
            self.a[i] = x
            weights = self.weights[i]
            act_fun = self.act_funs[i]
            z = np.dot(weights, x)
            self.z[i] = z
            if act_fun == 'sigmoid':
                a = fun.sigmoid(z)
            elif act_fun == 'ReLU':
                a = fun.ReLU(z)
            elif act_fun == 'tanh':
                a = fun.tanh(z)
            elif act_fun == 'softmax':
                a = fun.softmax(z)
            elif act_fun == 'ELU':
                a = fun.ELU(z)
            x = a
        return x

    def is_correct_prediction(self, output, label):
        '''
        Checks if the prediction of the network is correct.
        '''
        return np.argmax(output) == np.argmax(label)

    def backward(self, loss_grad, epoch, n):
        '''
        Propagates the loss gradient backwards through the net, accumulating
        in the weight changes for each layer.
        Params:
            loss_grad - np.array, the loss gradient wrt the input of the output layer
            epoch - int, the current training epoch
        '''
        for i in range(len(self.weights)-1, -1, -1):
            act_fun = self.act_funs[i]
            z = self.z[i]
            a = self.a[i]

            if act_fun == 'sigmoid':
                sigma_prime = fun.sigmoid_prime(z)
            elif act_fun == 'ReLU':
                sigma_prime = fun.ReLU_prime(z)
            elif act_fun == 'tanh':
                sigma_prime = fun.tanh_prime(z)
            elif act_fun == 'ELU':
                sigma_prime = fun.ELU_prime(z)


            if i == (len(self.weights)-1):
                error = loss_grad
            else:
                # Unselect bias weights from the previous layer
                weights = self.weights[i+1][:, :-1]
                term = np.dot(np.transpose(weights), error)
                error = np.multiply(term, sigma_prime)

            self.calc_weight_change(error, a, epoch, n, i)

    def calc_loss(self, output, label):
        '''
        Calculates the loss and loss gradient with respect to the input of the
        output layer.
        '''
        if self.loss_fun == 'cross_entropy' and self.act_funs[-1] == 'softmax':
            loss = fun.cross_entropy(output, label)
            loss_grad = fun.cross_entropy_prime(output, label)
        return loss, loss_grad

    def calc_weight_change(self, error, activation, epoch, n, i):
        '''
        Submethod of backward. Calculates the weight changes with annealing
        learn rate and L2 regularization.
        Params:
            error - np.array, the error of the weights in the layer
            activation - np.array, the activation of the previous layer
            epoch - int, the current training epoch
            n - int, batch size
            i - int, the index of the current layer
        '''
        self.grads[i] += (1/n) * np.outer(error, activation)
        learn_rate = self.alpha * (1 / (1 + epoch/self.T))
        weight_decay = self.gamma * self.weights[i]
        del_w = -(learn_rate/n) * (np.outer(error, activation) + weight_decay)
        self.del_weights[i] += del_w

    def update_weights(self):
        '''
        Updates the weights of the network.
        '''
        for i in range(len(self.del_weights)):
            self.del_weights[i] += self.mu * self.prev_del_weights[i]
            self.weights[i] += self.del_weights[i]
            self.prev_del_weights[i] = self.del_weights[i]
            self.del_weights[i] = np.zeros(shape=self.del_weights[i].shape)

    def validate(self, val_data):
        '''
        Validates the network.
        Params:
            val_data - tuple of np.arrays, the validation set
        '''
        loss_accu = 0
        corr_pred = 0
        n = len(val_data[0])
        for i in range(n):
            input = val_data[0][i]
            label = val_data[1][i]
            output = self.forward(input)
            loss, loss_grad = self.calc_loss(output, label)
            loss_accu += loss
            if self.is_correct_prediction(output, label):
                corr_pred += 1
        return loss_accu/n, corr_pred/n, self.weights

    def evaluate(self, data):
        '''
        Evaluates the network.
        Params:
            data - tuple of np.arrays, the data set used for model evaluation
        '''
        loss_accu = 0
        corr_pred = 0
        n = len(data[0])
        for i in range(n):
            input = data[0][i]
            label = data[1][i]
            output = self.forward(input)
            loss, loss_grad = self.calc_loss(output, label)
            loss_accu += loss
            if self.is_correct_prediction(output, label):
                corr_pred += 1
        return loss_accu/n, corr_pred/n

    def mini_batch_GD(self, epochs, batch_size, train_data, val_data, test_data,
                shuffle, augment, shift=2, stop=3, report=False, log=False):
        '''
        Performs mini batch gradient descent on the training data, validates
        the model on the validation data and evaluates the model on the test
        data.
        Params:
            epochs - int, the number of training epochs to be performed
            batch_size - int, the number of samples in the mini batches
            train_data - tuple of np.arrays, the training set
            val_data - tuple of np.arrays, the validation set
            test_data - tuple of np.arrays, the test set
            shuffle - boolean, triggers data shuffling
            augment - boolean, triggers data augmentation
            shift - int, data augmentation shift
            stop - int, # of increasing validation loss for early stopping
            report - boolean, turns on or off printout during training
            log - boolean, turns on or off logging during training
        '''
        if augment:
            train_data = self.augment_data(train_data, shift)
        for epoch in range(epochs):
            if shuffle:
                train_data = self.shuffle_data(train_data)
            batches = self.create_mini_batches(train_data, batch_size)
            for batch in batches:
                n = len(batch[0])
                for i in range(n):
                    input = batch[0][i]
                    label = batch[1][i]
                    output = self.forward(input)
                    loss, loss_grad = self.calc_loss(output, label)
                    self.backward(loss_grad, epoch, n)
                self.update_weights()
            val_loss, val_acc, weights = self.validate(val_data)
            train_loss, train_acc = self.evaluate(train_data)
            test_loss, test_acc = self.evaluate(test_data)

            if report:
                self.report(epoch, epochs, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc)
            if log:
                self.append_to_log(train_loss, val_loss, test_loss, train_acc, val_acc, test_acc)

            if stop > 0:
                if self.check_early_stopping(stop):
                    print('\nEarly stopping!')
                    break

    def report(self, epoch, epochs, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc):
        print('\nEpoch: {}/{}'.format(epoch+1, epochs))
        print('Training loss: ........{:3.4}'.format(train_loss))
        print('Training accuracy: ....{:3.4}'.format(train_acc))
        print('Validation loss: ......{:3.4}'.format(val_loss))
        print('Validation accuracy: ..{:3.4}'.format(val_acc))
        print('Test loss: ............{:3.4}'.format(test_loss))
        print('Test accuracy: ........{:3.4}'.format(test_acc))

    def append_to_log(self, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc):
        self.log['train_loss'].append(train_loss)
        self.log['val_loss'].append(val_loss)
        self.log['test_loss'].append(test_loss)
        self.log['train_acc'].append(train_acc)
        self.log['val_acc'].append(val_acc)
        self.log['test_acc'].append(test_acc)
        self.log['weights'].append(self.weights)

    def check_early_stopping(self, stop):
        val_loss = self.log['val_loss']
        weights = self.log['weights']
        if len(val_loss) > stop:
            n_increasing = 0
            prev_loss = np.inf
            for i in range(len(val_loss)):
                if val_loss[i] > prev_loss:
                    n_increasing += 1
                else:
                    n_increasing = 0
                if n_increasing == stop:
                    best = np.argmin(val_loss)
                    self.weights = weights[best]
                    return True
                prev_loss = val_loss[i]
        return False

    def augment_data(self, train_data, shift):
        aug_train_data = data.augment(train_data, shift)
        aug_X_train, aug_Y_train = aug_train_data
        X_train, Y_train = train_data
        X_train = np.concatenate((X_train, aug_X_train), axis=0)
        Y_train = np.concatenate((Y_train, aug_Y_train), axis=0)
        return (X_train, Y_train)

    def plot_accuracy(self):
        train_acc = self.log['train_acc']
        val_acc = self.log['val_acc']
        test_acc = self.log['test_acc']
        epochs = [(e+1) for e in range(len(train_acc))]
        plt.plot(epochs, train_acc, epochs, val_acc, epochs, test_acc)
        plt.xlabel('Training epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Training accuracy', 'Validation accuracy', 'Test accuracy'])
        plt.show()

    def plot_loss(self):
        train_loss = self.log['train_loss']
        val_loss = self.log['val_loss']
        test_loss = self.log['test_loss']
        epochs = [(e+1) for e in range(len(train_loss))]
        plt.plot(epochs, train_loss, epochs, val_loss, epochs, test_loss)
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')
        plt.legend(['Training loss', 'Validation loss', 'Test loss'])
        plt.show()

def main():
    test_mini_batch_gd = True
    test_numerical_grad = False

    # Data parameters
    train_size = 60000
    test_size = 10000
    val_frac = 0.1
    train_data, val_data, test_data, idx_to_class, class_to_idx = data.preprocess(train_size, test_size, val_frac)

    # Model parameters
    layers = [784, 32, 32, 10]
    act_funs = ['ELU', 'tanh','softmax']
    loss_fun = 'cross_entropy'
    alpha = 0.5
    mu = 0.4
    T = 80
    xavier = True
    p = 0
    gamma = 0
    network = MLP(layers, act_funs, loss_fun, alpha, mu, T, idx_to_class, xavier, p, gamma)

    if test_mini_batch_gd:
        # Training parameters
        epochs = 50
        batch_size = 128
        shuffle = True
        augment = False
        shift = 2
        stop = 3
        report = True
        log = True

        network.mini_batch_GD(epochs, batch_size, train_data, val_data, test_data, shuffle, augment, shift, stop, report, log)
        network.plot_accuracy()
        network.plot_loss()

    if test_numerical_grad:
        # Numerical grad parameters
        input, output = train_data[0][0], train_data[1][0]
        epsilon = 10**-2

        max_difference_1 = numerical_grad(network, 0, input, label, epsilon)
        max_difference_2 = numerical_grad(network, 1, input, label, epsilon)


if __name__ == '__main__':
    main()
