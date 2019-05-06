import torch
from torch import nn
from torchvision import models

def xavier_initialization(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.0)


def calc_num_ouput_features(image_height, image_width, num_filters, kernel_sizes, strides, paddings, poolings):
    '''
    Calculates the number of output features from the feature extractor.
    Args:
        image_height - int, image height
        image_width - int, image width
        num_filters - list of ints
        kernel_sizes - list of ints
        strides - list of ints
        paddings - list of ints
        poolings - list of booleans
    '''
    output_height = image_height
    output_width = image_width

    for i in range(len(num_filters)):
        output_height = (output_height - kernel_sizes[i] + 2*paddings[i]) // (strides[i]) + 1
        output_width = (output_width - kernel_sizes[i] + 2*paddings[i]) // (strides[i]) + 1
        if poolings[i]:
            output_height = output_height // 2
            output_width = output_width // 2

    num_output_features = output_height * output_width * num_filters[-1]
    return int(num_output_features)


def create_conv_module(in_channels, out_channels, kernel_size, stride, padding, pooling, dropout_prob, batch_norm):
    '''
    Creates a block consisting of a Conv2d layer, a ReLU layer and a MaxPool2d
    layer if pooling is True.
    Args:
        in_channels - int
        out_channels - int
        kernel_size - int
        stride - int
        padding - int
        pooling - boolean
        batch_norm - boolean
    '''
    block = nn.ModuleList()
    if batch_norm:
        block.extend([nn.BatchNorm2d(in_channels)])

    block.extend([nn.ReLU()])

    conv_layer = nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding
    )

    block.extend([conv_layer])

    if dropout_prob > 0:
        block.extend([nn.Dropout(p=dropout_prob)])

    if pooling:
        block.extend([nn.MaxPool2d(kernel_size=2, stride=2)])

    return block


def append_conv_module(layers, in_channels, out_channels, kernel_size, stride, padding, pooling, dropout_prob, batch_norm):
    module = create_conv_module(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pooling,
        dropout_prob,
        batch_norm
    )

    for layer in module:
        layers.extend([layer])


def create_classifier_module(n_in, n_out, batch_norm, dropout_prob, activation_fun=None):
    fc_layer = nn.Linear(n_in, n_out)
    module = nn.ModuleList([fc_layer])

    if batch_norm:
        module.extend([nn.BatchNorm1d(n_out)])

    if activation_fun == 'relu':
        module.extend([nn.ReLU()])
    elif activation_fun == 'sigmoid':
        module.extend([nn.Sigmoid()])
    elif activation_fun == 'tanh':
        module.extend([nn.Tanh()])
    elif activation_fun == 'softmax':
        module.extend([nn.Softmax()])

    if dropout_prob > 0:
        module.extend([nn.Dropout(p=dropout_prob)])

    return module


def append_classifier_module(layers, n_in, n_out, batch_norm, dropout_prob, activation_fun=None):
    module = create_classifier_module(
        n_in,
        n_out,
        batch_norm,
        dropout_prob,
        activation_fun
    )

    for layer in module:
        layers.extend([layer])


class LeNet(nn.Module):
    def __init__(self, image_channels, num_classes):
        '''
        Class for the LeNet CNN architecture used for image classification.
        Args:
            image_channels - int, the number of image channels
            num_classes - int, the number of classification classes
            num_filters - list of ints, the number of filters in the CNN
        '''
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels = image_channels,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        num_output_features = 4*4*128
        self. num_output_features = num_output_features
        self.classifier = Classifier(
            num_input_features=num_output_features,
            num_classes=num_classes,
            num_hidden_units=[64],
            activation_funs=['relu'],
            dropout_prob=0,
            batch_norm=False
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        return x


class Classifier(nn.Module):
    def __init__(self,
                num_input_features,
                num_classes,
                num_hidden_units,
                activation_funs,
                dropout_prob,
                batch_norm):
        '''
        Fully connected classifier network.
        Args:
            num_input_features - int, the number of input features
            num_classes - int, the number of classes
            num_hidden_units - list of ints, the number of hidden units
            activation_funs - list of strings, the activation functions
            dropout_prob - float, dropout probability
            batch_norm - boolean, triggers batch normalization
        '''

        if len(num_hidden_units) != len(activation_funs):
            raise ValueError(
                'The number of hidden units and activation functions must \
                be equal!'
            )

        super().__init__()

        self.layers = nn.ModuleList()

        # Initial layer
        append_classifier_module(
            self.layers,
            n_in=num_input_features,
            n_out=num_hidden_units[0],
            batch_norm=batch_norm,
            dropout_prob=dropout_prob,
            activation_fun=activation_funs[0]
        )

        # Hidder layers
        if len(num_hidden_units) > 1:
            for n_in, n_out, act_fun in zip(num_hidden_units[:-1], num_hidden_units[1:], activation_funs[1:]):
                append_classifier_module(
                    self.layers,
                    n_in=n_in,
                    n_out=n_out,
                    batch_norm=batch_norm,
                    dropout_prob=dropout_prob,
                    activation_fun=act_fun
                )

        # Output layer
        append_classifier_module(
            self.layers,
            n_in=num_hidden_units[-1],
            n_out=num_classes,
            batch_norm=False,
            dropout_prob=0,
            activation_fun=None
        )

        self.requires_grad = True

    def forward(self, x):
        '''
        Feedforward method.
        Args:
            x - torch.Tensor, the input feature tensor
        '''
        for layer in self.layers:
            x = layer(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self,
                num_image_channels,
                num_filters,
                kernel_sizes,
                strides,
                paddings,
                poolings,
                dropout_prob,
                batch_norm):
        '''
        Fully connected classifier networks.
        Args:
            num_image_channels - int
            num_filters - list of int, the number of filters for each conv block
            kernel_sizes - list of ints
            strides - list of ints
            paddings - list of ints
            poolings - list of booleans
            dropout_prob - float
            batch_norm - boolean
        '''
        super().__init__()

        self.layers = nn.ModuleList()

        # First convolutional module
        append_conv_module(
            layers = self.layers,
            in_channels = num_image_channels,
            out_channels = num_filters[0],
            kernel_size = kernel_sizes[0],
            stride = strides[0],
            padding = paddings[0],
            pooling = poolings[0],
            dropout_prob = dropout_prob,
            batch_norm = batch_norm[0]
        )

        # Consequtive convolutional modules
        for i in range(1,len(num_filters)):
            append_conv_module(
                layers = self.layers,
                in_channels = num_filters[i-1],
                out_channels = num_filters[i],
                kernel_size = kernel_sizes[i],
                stride = strides[i],
                padding = paddings[i],
                pooling = poolings[i],
                dropout_prob = dropout_prob,
                batch_norm = batch_norm[i]
            )

        self.requires_grad = True

    def forward(self, x):
        '''
        Feedforward method.
        Args:
            x - torch.Tensor, the input tensor
        '''
        for layer in self.layers:
            x = layer(x)
        return x


class MyCNN(nn.Module):
    '''
    CNN class. Uses the FeatureExtractor class for the feature extraction
    module and the Classifier class for the classification module.
    Args:
        num_image_channels - int, the number of image channels
        num_filters -
        kernel_size -
        stride - int, the stride of the convolutional layers
        padding - int, the padding of the convolutional layers
        pooling - boolean, triggers maxpooling
        batch_norm - boolean, triggers batch_norm
        xavier - boolean, triggers Xavier weight initialization
        num_classes - int, the number of classes
        num_hidden_units - list of ints, the number of hidden units in the
        classifier network
        activation_funs - list of strings, the activation functions of the
        classifier network
        dropout_prob - float, the dropout probability of the classifier network
    '''
    def __init__(self,
        image_height,
        image_width,
        num_image_channels,
        num_filters,
        kernel_sizes,
        strides,
        paddings,
        poolings,
        feat_dropout_prob,
        feat_batch_norm,
        xavier,
        num_classes,
        num_hidden_units,
        activation_funs,
        class_dropout_prob,
        class_batch_norm):

        super().__init__()

        self.feature_extractor = FeatureExtractor(
            num_image_channels = num_image_channels,
            num_filters = num_filters,
            kernel_sizes = kernel_sizes,
            strides = strides,
            paddings = paddings,
            poolings = poolings,
            dropout_prob = feat_dropout_prob,
            batch_norm = feat_batch_norm
        )

        num_output_features = calc_num_ouput_features(
            image_height = image_height,
            image_width = image_width,
            num_filters = num_filters,
            kernel_sizes = kernel_sizes,
            strides = strides,
            paddings = paddings,
            poolings = poolings
        )

        self.num_output_features = num_output_features

        self.classifier = Classifier(
            num_input_features = num_output_features,
            num_classes = num_classes,
            num_hidden_units = num_hidden_units,
            activation_funs = activation_funs,
            dropout_prob = class_dropout_prob,
            batch_norm = class_batch_norm,
        )

        if xavier:
            self.feature_extractor.apply(xavier_initialization)
            self.classifier.apply(xavier_initialization)

    def forward(self, x):
        """
        Feedforward through the network.
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        return x


class TransferLearner(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features*4, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        '''
        Feedforward through the network.
        Args:
            x - torch.Tensor, the input image
        '''
        x = nn.functional.interpolate(x, scale_factor=8)
        x = self.model(x)
        return x
