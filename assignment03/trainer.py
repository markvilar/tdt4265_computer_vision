import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy
from torchvision import models
import networks
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def save_to_file(trainer):
    train_loss = np.array(trainer.TRAIN_LOSS)
    validation_loss = np.array(trainer.VALIDATION_LOSS)
    test_loss = np.array(trainer.TEST_LOSS)
    train_acc = np.array(trainer.TRAIN_ACC)
    validation_acc = np.array(trainer.VALIDATION_ACC)
    test_acc = np.array(trainer.TEST_ACC)

    data = np.array([
                        train_loss,
                        validation_loss,
                        test_loss,
                        train_acc,
                        validation_acc,
                        test_acc
                    ])

    filename = trainer.prefix + '_data_log.csv'
    data.tofile(filename, sep=',', format='%10.5f')


def extract_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    
    n = int(len(data) / 6)

    train_loss = data[0:n]
    validation_loss = data[n:2*n]
    test_loss = data[2*n:3*n]
    train_acc = data[3*n:4*n]
    validation_acc = data[4*n:5*n]
    test_acc = data[5*n:6*n]

    i = np.argmin(validation_loss)
    
    print('Final train loss: {}'.format(train_loss[i]))
    print('Final validation loss: {}'.format(validation_loss[i]))
    print('Final test loss: {}'.format(test_loss[i]))
    print('Final train accuracy: {}'.format(train_acc[i]))
    print('Final validation accuracy: {}'.format(validation_acc[i]))
    print('Final test accuracy: {}'.format(test_acc[i]))


def comparison_plot(filename1, filename2, legend1, legend2):
    # Validation and training loss
    data1 = np.genfromtxt(filename1, delimiter=',')
    n1 = int(len(data1) / 6)
    train_loss1 = data1[0:n1]
    validation_loss1 = data1[n1:2*n1]

    data2 = np.genfromtxt(filename2, delimiter=',')
    n2 = int(len(data2) / 6)
    train_loss2 = data2[0:n2]
    validation_loss2 = data2[n2:2*n2]
    
    train_loss_legend1 = legend1 + ' training loss'
    train_loss_legend2 = legend2 + ' training loss'
    validation_loss_legend1 = legend1 + ' validation loss'
    validation_loss_legend2 = legend2 + ' validation loss'

    plt.figure(figsize=(12, 8))
    plt.title("Training and validation loss")
    plt.plot(train_loss1, label=train_loss_legend1)
    plt.plot(train_loss2, label=train_loss_legend2)
    plt.plot(validation_loss1, label=validation_loss_legend1)
    plt.plot(validation_loss2, label=validation_loss_legend2)
    plt.legend()
    plt.savefig(os.path.join("plots", "loss_comparison.png"))
    plt.show()


class Visualizer:
    def __init__(self):
        '''
        '''
        self.model = models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.first_layer = self.model.conv1
        self.last_layer = self.model.layer4[1].conv2
        self.dataloader_train, a, b = load_cifar10(1)

        #x = nn.functional.interpolate(x, scale_factor=8)
        #x = self.first_layer(x)

    def visualize(self):
        os.makedirs("images", exist_ok=True)
        tensor = next(iter(self.dataloader_train))[0]
        tensor = nn.functional.interpolate(tensor, scale_factor=8)
        
        self.save_image(tensor)

        self.visualize_first_layer(tensor)
        self.visualize_last_layer(tensor)

    def save_image(self, tensor):
        array = tensor.data[0].cpu().numpy()
        array = array.transpose([1,2,0])
        matplotlib.image.imsave('images/original.png', array)


    def visualize_first_layer(self, tensor):
        filtered_tensor = self.first_layer(tensor)
        for i in range(10):
            filename = 'images/first_layer_' + str(i) + '.png'
            array = filtered_tensor[0][i].cpu().numpy()
            matplotlib.image.imsave(filename, array)


    def visualize_last_layer(self, tensor):
        filtered_tensor = self.last_layer(tensor)
        for i in range(10):
            filename = 'images/last_layer_' + str(i) + '.png'
            array = filtered_tensor[0][i].cpu().numpy()
            matplotlib.image.imsave(filename, array)



class Trainer:
    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 10 
        self.batch_size = 32
        self.learning_rate = 5e-3
        self.early_stop_count = 4
        self.weight_decay = 0
        optimizer = 'adam'

        # Architecture
        image_height = 32
        image_width = 32
        num_image_channels = 3
        num_filters = [96, 80, 96, 64]
        kernel_sizes = [5, 5, 5, 5]
        strides = [1, 1, 1, 1]
        paddings = [2, 2, 2, 2]
        poolings = [True, True, False, False]
        feat_dropout_prob = 0
        feat_batch_norm = [True, True, True, True]
        xavier = True
        num_classes = 10
        num_hidden_units = [64]
        activation_funs = ['relu']
        class_dropout_prob = 0
        class_batch_norm = True

        model = 'my_cnn'

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()

        # Initialize the mode
        if model == 'lenet':
            self.model = networks.LeNet(
                num_image_channels,
                num_classes
            )
            self.prefix = 'lenet'
        elif model == 'my_cnn':
            self.model = networks.MyCNN(
                image_height=image_height,
                image_width=image_width,
                num_image_channels=num_image_channels,
                num_filters=num_filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                poolings=poolings,
                feat_dropout_prob=feat_dropout_prob,
                feat_batch_norm=feat_batch_norm,
                xavier=xavier,
                num_classes=num_classes,
                num_hidden_units=num_hidden_units,
                activation_funs=activation_funs,
                class_dropout_prob=class_dropout_prob,
                class_batch_norm=class_batch_norm
            )
            self.prefix = 'my_cnn'
        elif model == 'transfer_learner':
            self.model = networks.TransferLearner(num_classes)
            self.prefix = 'transfer_learner'

        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train)

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)

                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()

                 # Compute loss/accuracy for all three datasets.
                if (batch_it + 1) % self.validation_check == 0:
                    self.validation_epoch()
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        return


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()

    print("Final training accuracy:", trainer.TRAIN_ACC[-trainer.early_stop_count])
    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])

    print("Final training loss:", trainer.TRAIN_LOSS[-trainer.early_stop_count])
    print("Final test loss:", trainer.TEST_LOSS[-trainer.early_stop_count])
    print("Final validation loss:", trainer.VALIDATION_LOSS[-trainer.early_stop_count])
    
    save_to_file(trainer)
