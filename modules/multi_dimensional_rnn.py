""" This class will implement Multi-dimensional RNN modules and also Multi-dimensional LSTM modules
    These will be based on a generalization of the native pytorch RNN code
    from:  https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
"""

import math
import torch
import warnings
import itertools
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn
from util.image_input_transformer import ImageInputTransformer
import data_preprocessing.load_mnist
import torch.nn as nn
import torchvision.transforms as transforms


class MDRNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, h1, h2, hidden_label=''):
        # if input.size(0) != hx.size(0):
        #   raise RuntimeError(
        #       "Input batch size {} doesn't match hidden{} batch size {}".format(
        #           input.size(0), hidden_label, hx.size(0)))

        # TODO: Fix the first check

        if h1.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, h1.size(1), self.hidden_size))

        if h1.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, h2.size(1), self.hidden_size))


class MDRNNCell(MDRNNCellBase):
    r"""A (2D) Multi-Dimensional RNN cell with tanh or ReLU non-linearity.
      .. math::
          Normal 1D RNN:
          h' = \tanh(w_{ih} * x + b_{ih}  +  w_{hh} * h + b_{hh})
          2D RNN:
          h' = \tanh(w_{ih} * x + b_{ih}  +
                        w_{hh} * h + b_{hh}  +
                        w_{hh} * h + b_{hh}  +
                )

      If nonlinearity='relu', then ReLU is used in place of tanh.
      Args:
          input_size: The number of expected features in the input x
          hidden_size: The number of features in the hidden states h1 and h2
          bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
              Default: ``True``
          nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
      Inputs: input, hidden1, hidden2
          - **input** (batch, input_size): tensor containing input features
          - **hidden** (batch, hidden_size): tensor containing the initial hidden1
            state for each element in the batch.
          - **hidden2** (batch, hidden_size): tensor containing the initial hidden2
            state for each element in the batch.
      Outputs: h'
          - **h'** (batch, hidden_size): tensor containing the next hidden state
            for each element in the batch
      Attributes:
          weight_ih: the learnable input-hidden1 weights, of shape
              `(input_size x hidden_size)`
          weight_h1h: the learnable hidden1-hidden weights, of shape
              `(hidden_size x hidden_size)`
          weight_hh2: the learnable hidden2-hidden weights, of shape
              `(hidden_size x hidden_size)`
          bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
          bias_h1h: the learnable hidden1-hidden bias, of shape `(hidden_size)`
          bias_h2h: the learnable hidden2-hidden bias, of shape `(hidden_size)`
      Examples::
          >>> rnn = nn.RNNCell(10, 20)
          >>> input = Variable(torch.randn(6, 3, 10))
          >>> h1 = Variable(torch.randn(3, 20))
          >>> h2 = Variable(torch.randn(3, 20))
          >>> output = []
          >>> for i in range(6):
          ...     hx = rnn(input[i], h1, h2)
          ...     output.append(hx)
      """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(MDRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.ih = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h1h = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

        if not bias:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_h1h', None)
            self.register_parameter('bias_h2h', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h1, h2):
        self.check_forward_input(input)
        self.check_forward_hidden(input, h1, h2)
        if self.nonlinearity == "tanh":
            func = F.tanh
        elif self.nonlinearity == "relu":
            # func = self._backend.RNNReLUCell
            func = F.relu
        elif self.nonlinearity == "sigmoid":
            func = F.sigmoid
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        input_total = self.ih(input)
        h1h_total = self.h1h(h1)
        h2h_total = self.h1h(h2)
        # h_total = torch.add
        # total = torch.add(h_total, 1, input_total)
        total = input_total + h1h_total + h2h_total
        h_activation = func(total)
        # h_activation = torch.sigmoid(total)

        return h_activation

        # return func(
        #    input, hx,
        #    self.weight_ih, self.weight_hh,
        #    self.bias_ih, self.bias_hh,
        # )


class MultiDimensionalRNN(torch.nn.Module):
    def __init__(self, nonlinearity="tanh"):
        super(MultiDimensionalRNN, self).__init__()
        self.input_channels = 1
        self.output_channels = 1
        self.state_convolution = nn.Conv1d(self.input_channels,
                                           self.output_channels, 2,
                                           padding=1)
        self.input_convolution = nn.Conv2d(self.input_channels,
                                           self.output_channels, 1)
        self.fc3 = nn.Linear(1024, 10)
        self.nonlinearity = nonlinearity

    @staticmethod
    def create_multi_dimensional_rnn(nonlinearity="tanh"):
        return MultiDimensionalRNN(nonlinearity)

    def get_activation_function(self):
        if self.nonlinearity == "tanh":
            activation_function = F.tanh
        elif self.nonlinearity == "relu":
            # func = self._backend.RNNReLUCell
            activation_function = F.relu
        elif self.nonlinearity == "sigmoid":
            activation_function = F.sigmoid
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return activation_function

    # Input tensor x is an image
    def forward(self, x):
        # Step 1: Create a skewed version of the input image
        #skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(x)
        skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensors(x)

        image_height = x.size(1)
        previous_state_column = Variable(torch.zeros(self.input_channels,
                                                     self.output_channels,
                                                     image_height))
        # print("image_height: " + str(image_height))
        original_image_columns = x.size(2)
        skewed_image_rows = skewed_image.size(1)
        skewed_image_columns = skewed_image.size(2)
        #print("skewed image columns: " + str(skewed_image_columns))
        #print("skewed image rows: " + str(skewed_image_rows))

        print("skewed_image: " + str(skewed_image))

        # The image is 3-dimensional, but the convolution somehow
        # requires 4-dimensional input (why?) Seems pretty odd, but see also
        # http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        # http://pytorch.org/docs/master/torch.html#torch.unsqueeze
        skewed_image_four_dim = torch.unsqueeze(skewed_image, 0)
        skewed_image_variable = Variable(skewed_image_four_dim)
        input_matrix = self.input_convolution(skewed_image_variable)
        # print("input_matrix: " + str(input_matrix))

        # Need to somehow initialize the hidden states column
        state_column = None

        activations = list([])

        for column_number in range(0, skewed_image_columns):
            # Compute convolution on previous state column vector padded with zeros
            state_column_both_sides_padding = self.state_convolution(previous_state_column)
            # Throw away the last element, which comes from padding on the bottom, as
            # there seems to be no way in pytorch to get the padding only on one side
            state_column = state_column_both_sides_padding[:, :, 0:image_height]
            # print("state_column.size(): " + str(state_column.size()))

            input_column = input_matrix[:, :, :, column_number]
            state_plus_input = state_column + input_column
            # print("input_column: " + str(input_column))
            # print("state_plus_input: " + str(state_plus_input))
            activation_column = self.get_activation_function()(state_plus_input)
            # print("activation: " + str(activation_column))
            previous_state_column = activation_column
            activations.append(activation_column)

        # How to unskew the activation matrix, and retrieve an activation
        # matrix of the original image size?
        activations_column_transposed = torch.transpose(activations[0], 1, 2)
        activations_as_tensor = activations_column_transposed
        for column_number in range(1, skewed_image_columns):
            activations_column_transposed = torch.transpose(activations[column_number], 1, 2)
            activations_as_tensor = torch.cat((activations_as_tensor, activations_column_transposed), 2)
        #print("activations_as_tensor: " + str(activations_as_tensor))

        activations_unskewed = activations_as_tensor[:, 0, 0:original_image_columns]
        #print("activations_unskewed:" + str(activations_unskewed))
        for row_number in range(1, skewed_image_rows):
            activations = activations_as_tensor[:, row_number, row_number: (original_image_columns + row_number)]
            activations_unskewed = torch.cat((activations_unskewed,activations), 0)
        #print("activations_unskewed: " + str(activations_unskewed))

        #return activations_unskewed
        activations_one_dimensional = activations_unskewed.view(-1, 1024)
        #print("activations_one_dimensional: " + str(activations_one_dimensional))
        # It is nescessary to output a tensor of size 10, for 10 different output classes
        result = self.fc3(activations_one_dimensional)
        return result


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_mdrnn_cell():
    print("Testing the MultDimensionalRNN Cell... ")
    mdrnn = MDRNNCell(10, 5, nonlinearity="relu")
    input = Variable(torch.randn(6, 3, 10))

    # print("Input: " + str(input))

    h1 = Variable(torch.randn(3, 5))
    h2 = Variable(torch.randn(3, 5))
    output = []

    for i in range(6):
        print("iteration: " + str(i))
        h2 = mdrnn(input[i], h1, h2)
        print("h2: " + str(h2))
        output.append(h2)

    print(str(output))


def test_mdrnn():
    image = data_preprocessing.load_mnist.get_first_image()
    multi_dimensional_rnn = MultiDimensionalRNN.create_multi_dimensional_rnn(nonlinearity="sigmoid")
    multi_dimensional_rnn.forward(image)


def train_mdrnn():
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    multi_dimensional_rnn = MultiDimensionalRNN.create_multi_dimensional_rnn(nonlinearity="sigmoid")
    #multi_dimensional_rnn = Net()
    optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.001, momentum=0.9)

    trainloader = data_preprocessing.load_mnist.get_train_loader()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # See: https://stackoverflow.com/questions/48015235/i-get-this-error-on-pytorch-runtimeerror-invalid-argument-2-size-1-x-400?rq=1
            #LRTrans = transforms.Compose(
            #    [transforms.Scale((32, 32)),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            #inputs = LRTrans(inputs)

            # wrap them in Variable
            labels = Variable(labels)
            #labels, inputs = Variable(labels), Variable(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            #print("inputs: " + str(inputs))


            # forward + backward + optimize
            outputs = multi_dimensional_rnn(inputs)
            #print("outputs: " + str(outputs))
            #print("labels: " + str(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def main():
    # test_mdrnn_cell()
    #test_mdrnn()
    train_mdrnn()


if __name__ == "__main__":
    main()
