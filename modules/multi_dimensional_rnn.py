""" This class will implement Multi-dimensional RNN modules and also Multi-dimensional LSTM modules
    These will be based on a generalization of the native pytorch RNN code
    from:  https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
"""

import math
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn
from util.image_input_transformer import ImageInputTransformer
import torch.nn as nn
import time
import util.tensor_flipping
from abc import ABCMeta, abstractmethod


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


class MultiDimensionalRNNBase(torch.nn.Module):
    def __init__(self, hidden_states_size: int,
                 batch_size,  compute_multi_directional: bool,
                 nonlinearity="tanh",):
        super(MultiDimensionalRNNBase, self).__init__()

        self.batch_size = batch_size
        self.nonlinearity = nonlinearity
        self.input_channels = 1
        self.hidden_states_size = hidden_states_size
        self.selection_tensor = Variable(self.create_torch_indices_selection_tensor(batch_size))
        if MultiDimensionalRNNBase.use_cuda():
            self.selection_tensor = self.selection_tensor.cuda()
        self.compute_multi_directional = compute_multi_directional

    # This function is slow because all four function calls for 4 directions are
    # executed sequentially. It isn't entirely clear how to optimize this.
    # See the discussion at:
    # https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360
    def forward_multi_directional_multi_dimensional_function_fast(self, x):
        # print("list(x.size()): " + str(list(x.size())))

        x_direction_two = util.tensor_flipping.flip(x, 2)
        x_direction_three = util.tensor_flipping.flip(x, 3)
        x_direction_four = util.tensor_flipping.flip(util.tensor_flipping.flip(x, 2), 3)
        x_multiple_directions = torch.cat((x, x_direction_two, x_direction_three, x_direction_four), 0)

        number_of_examples = x.size(0)
        # print("number of examples: " + str(number_of_examples))

        # Original order
        activations_unskewed = self._compute_multi_dimensional_function_one_direction(x_multiple_directions)
        if number_of_examples == self.batch_size:
            selection_tensor = self.selection_tensor
        else:
            selection_tensor = Variable(MultiDimensionalRNNBase.create_torch_indices_selection_tensor(
                number_of_examples))
            if MultiDimensionalRNNBase.use_cuda():
                selection_tensor = selection_tensor.cuda()

        # print("activations_unskewed: " + str(activations_unskewed))
        # print("selection_tensor: " + str(selection_tensor))

        # Using tor.index_select we can bring together the activations of the four
        # different rotations, while avoiding use of a for loop, making the whole thing
        # hopefully faster
        activations_selected = torch.index_select(activations_unskewed, 0, selection_tensor)
        # print("activations_selected: " + str(activations_selected))

        # activations_rearranged = torch.cat((activations_unskewed[0, :, :],
        #                                     activations_unskewed[number_of_examples, :, :],
        #                                     activations_unskewed[number_of_examples * 2, :, :],
        #                                     activations_unskewed[number_of_examples * 3, :, :],), 0)
        # activations_rearranged = activations_rearranged.unsqueeze(0)
        # print("activations_rearranged: " + str(activations_rearranged))
        # for i in range(1, number_of_examples):
        #    activations_rearranged_row = torch.cat((activations_unskewed[i, :, :],
        #                                         activations_unskewed[number_of_examples + 1, :, :],
        #                                         activations_unskewed[number_of_examples * 2 + i, :, :],
        #                                         activations_unskewed[number_of_examples * 3 + i, :, :],), 0)
        #    activations_rearranged_row = activations_rearranged_row.unsqueeze(0)
        #    activations_rearranged = torch.cat((activations_rearranged, activations_rearranged_row), 0)

        # print("activations_rearranged: " + str(activations_rearranged))
        activations_one_dimensional = activations_selected.view(-1, self.number_of_output_dimensions())
        # activations_one_dimensional = activations_rearranged.view(-1, 32 * 32 * 4)

        # print("activations_combined: " + str(activations_combined))

        # print("activations_one_dimensional: " + str(activations_one_dimensional))
        # It is nescessary to output a tensor of size 10, for 10 different output classes
        result = self._final_activation_function(activations_one_dimensional)
        return result

    # Needs to be implemented in the subclasses
    @abstractmethod
    def _final_activation_function(self, final_activation_function_input):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def _compute_multi_dimensional_function_one_direction(self, function_input):
        raise RuntimeError("not implemented")

    @staticmethod
    def use_cuda():
        return torch.cuda.is_available()

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

    @staticmethod
    def create_indices_list(number_of_examples):
        result = []
        for i in range(0, number_of_examples):
            result.append(i)
            result.append(number_of_examples + i)
            result.append(number_of_examples * 2 + i)
            result.append(number_of_examples * 3 + i)
        return result

    @staticmethod
    def create_torch_indices_selection_tensor(number_of_examples):
        indices = MultiDimensionalRNNBase.create_indices_list(number_of_examples)
        result = torch.LongTensor(indices)
        return result

    @staticmethod
    def create_skewed_images_variable_four_dim(x):
        skewed_images = ImageInputTransformer.create_row_diagonal_offset_tensors(x)

        # print("skewed images columns: " + str(skewed_images_columns))
        # print("skewed images rows: " + str(skewed_images_rows))
        # print("skewed_images: " + str(skewed_images))

        skewed_images_four_dim = torch.unsqueeze(skewed_images, 1)
        skewed_images_variable = Variable(skewed_images_four_dim)
        if MultiDimensionalRNNBase.use_cuda():
            skewed_images_variable = skewed_images_variable.cuda()
        return skewed_images_variable


    @staticmethod
    def extract_unskewed_activations(activations,
                                     original_image_columns: int,
                                     skewed_image_columns: int,
                                     skewed_image_rows: int):

        # print("original image columns: " + str(original_image_columns))

        # How to unskew the activation matrix, and retrieve an activation
        # matrix of the original image size?
        activations_column = activations[0]
        # Columns will be horizontally concatenated, add extra dimension for this concatenation
        activations_column_unsqueezed = torch.unsqueeze(activations_column, 3)
        activations_as_tensor = activations_column_unsqueezed
        for column_number in range(1, skewed_image_columns):
            # print("activations[column_number]: " + str(activations[column_number]))
            activations_column = activations[column_number]
            # print("activations column: " + str(activations_column))
            activations_column_unsqueezed = torch.unsqueeze(activations_column, 3)
            activations_as_tensor = torch.cat((activations_as_tensor, activations_column_unsqueezed), 3)
        # print("activations_as_tensor: " + str(activations_as_tensor))

        activations_unskewed = activations_as_tensor[:, :, 0, 0:original_image_columns]
        activations_unskewed = torch.unsqueeze(activations_unskewed, 2)
        # print("activations_unskewed before:" + str(activations_unskewed))
        for row_number in range(1, skewed_image_rows):
            activations = activations_as_tensor[:, :, row_number, row_number: (original_image_columns + row_number)]
            activations = torch.unsqueeze(activations, 2)
            # print("activations:" + str(activations))
            activations_unskewed = torch.cat((activations_unskewed, activations), 2)
        # print("activations_unskewed: " + str(activations_unskewed))
        return activations_unskewed

    @staticmethod
    def compute_state_convolution_and_remove_bottom_padding(convolution_layer,
                                                            previous_state_column,
                                                            image_height):
        # print("previous_state_column: " + str(previous_state_column))

        # Compute convolution on previous state column vector padded with zeros
        state_column_both_sides_padding = convolution_layer(previous_state_column)
        # Throw away the last element, which comes from padding on the bottom, as
        # there seems to be no way in pytorch to get the padding only on one side
        state_column = state_column_both_sides_padding[:, :, 0:image_height]
        return state_column

    @staticmethod
    def compute_states_plus_input(input_matrix, column_number, state_columns_combined):
        input_column = input_matrix[:, :, :, column_number]
        state_plus_input = state_columns_combined + input_column
        # print("input_column: " + str(input_column))
        # print("state_plus_input: " + str(state_plus_input))
        return state_plus_input

    def number_of_output_dimensions(self):
        result = 1024 * self.hidden_states_size
        if self.compute_multi_directional:
            result = result * 4
        return result


class StateUpdateBlock():
    def __init__(self, hidden_states_size: int):
        self.hidden_states_size = hidden_states_size
        self.state_one_convolution = nn.Conv1d(self.hidden_states_size,
                                                      self.hidden_states_size, 1)
        self.state_two_convolution = nn.Conv1d(self.hidden_states_size,
                                                      self.hidden_states_size, 1)

    @staticmethod
    def get_previous_state_column_static(previous_state_column, state_index: int,
                                         hidden_states_size: int):
        # print("previous memory state column: " + str(previous_memory_state_column))
        if state_index == 100:
            previous_memory_state_column_shifted = previous_state_column.clone()
            height = previous_state_column.size(2)
            zeros_padding = Variable(torch.zeros(previous_state_column.size(0), hidden_states_size, 1))
            if MultiDimensionalRNNBase.use_cuda():
                zeros_padding = zeros_padding.cuda()
            skip_first_sub_tensor = previous_memory_state_column_shifted[:, :, 0:(height - 1)]
            # print("zeros padding" + str(zeros_padding))
            # print("skip_first_sub_tensor: " + str(skip_first_sub_tensor))
            previous_memory_state_column_shifted = torch. \
                cat((zeros_padding, skip_first_sub_tensor), 2)
            # print("Returning previous_memory_state_column_shifted: " + str(previous_memory_state_column_shifted))
            return previous_memory_state_column_shifted
        return previous_state_column

    def get_previous_state_column(self, previous_state_column, state_index: int):
        return StateUpdateBlock.get_previous_state_column_static(previous_state_column,
                                                                 state_index,
                                                                 self.hidden_states_size)

    @staticmethod
    def compute_weighted_state_input_static(state_convolution, previous_state_column,
                                            state_index: int, hidden_states_size):
        return state_convolution(StateUpdateBlock.
                                 get_previous_state_column_static(previous_state_column, state_index,
                                                                  hidden_states_size))

    def compute_weighted_state_input(self, state_convolution, previous_state_column,
                                     state_index: int):
        return state_convolution(self.get_previous_state_column(previous_state_column, state_index))

    def compute_weighted_states_input(self, previous_state_column):
        state_one_result = self.state_one_convolution(self.get_previous_state_column(previous_state_column, 1))
        state_two_result = self.state_two_convolution(self.get_previous_state_column(previous_state_column, 2))
        result = state_one_result + state_two_result
        return result

    def get_state_convolutions_as_list(self):
        return [self.state_one_convolution, self.state_two_convolution]

class MultiDimensionalRNN(MultiDimensionalRNNBase):
    def __init__(self, hidden_states_size, batch_size, compute_multi_directional: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalRNN, self).__init__(hidden_states_size, batch_size,
                                                  compute_multi_directional,
                                                  nonlinearity)
        self.state_update_block = StateUpdateBlock(hidden_states_size)
        # This is necessary to make sure things are stored on the same gpu, otherwise
        # pytorch doesn't realizes these convolutions are part of this module
        self.state_convolutions = nn.ModuleList(self.state_update_block.get_state_convolutions_as_list())

        self.input_convolution = nn.Conv2d(self.input_channels,
                                           self.hidden_states_size, 1)
        # self.fc3 = nn.Linear(1024, 10)
        # For multi-directional rnn
        if self.compute_multi_directional:
            self.fc3 = nn.Linear(self.number_of_output_dimensions(), 10)
        else:
            self.fc3 = nn.Linear(self.number_of_output_dimensions(), 10)



    @staticmethod
    def create_multi_dimensional_rnn(hidden_states_size: int, batch_size: int,  compute_multi_directional: bool,
                                     nonlinearity="tanh"):
        return MultiDimensionalRNN(hidden_states_size, batch_size, compute_multi_directional,
                                   nonlinearity)

    def compute_multi_dimensional_rnn_one_direction(self, x):
        # Step 1: Create a skewed version of the input image
        # skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(x)

        # The image is 3-dimensional, but the convolution somehow
        # requires 4-dimensional input. Whdevice=gpus[0])y? This is probably because
        # one extra dimension is for channels, and the fourth dimension is for
        # doing multiple examples in a mini-batch in parallel
        # http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        # http://pytorch.org/docs/master/torch.html#torch.unsqueeze
        skewed_images_variable = MultiDimensionalRNNBase.create_skewed_images_variable_four_dim(x)
        image_height = x.size(2)
        number_of_examples = x.size(0)
        # print("image height: " + str(image_height))
        previous_hidden_state_column = Variable(torch.zeros(self.input_channels,
                                                            self.hidden_states_size,
                                                            image_height))
        if MultiDimensionalRNNBase.use_cuda():
            previous_hidden_state_column = previous_hidden_state_column.cuda()

        # print("image_height: " + str(image_height))

        skewed_image_columns = skewed_images_variable.size(3)

        input_matrix = self.input_convolution(skewed_images_variable)
        # print("input_matrix: " + str(input_matrix))

        activations = list([])

        for column_number in range(0, skewed_image_columns):
            # Compute convolution on previous state column vector padded with zeros
            state_columns_combined = self.state_update_block.compute_weighted_states_input(previous_hidden_state_column)

            # print("state_column.size(): " + str(state_column.size()))
            state_plus_input = MultiDimensionalRNNBase.compute_states_plus_input(input_matrix,
                                                                                 column_number,
                                                                                 state_columns_combined)
            activation_column = self.get_activation_function()(state_plus_input)
            #print("activation column: " + str(activation_column))
            previous_hidden_state_column = activation_column
            activations.append(activation_column)
            #print("activations length: " + str(len(activations)))

        original_image_columns = x.size(2)
        skewed_image_rows = skewed_images_variable.size(2)
        #print("Skewed image rows: " + str(skewed_image_rows))

        #print("activations: " + str(activations))
        activations_unskewed = MultiDimensionalRNNBase.\
            extract_unskewed_activations(activations, original_image_columns,
                                         skewed_image_columns, skewed_image_rows)
        #print("activations_unskewed: " + str(activations_unskewed))
        return activations_unskewed

    def forward_one_directional_multi_dimensional_rnn(self, x):
        activations_unskewed = self.compute_multi_dimensional_rnn_one_direction(x)
        activations_one_dimensional = activations_unskewed.view(-1, self.number_of_output_dimensions())
        # print("activations_one_dimensional: " + str(activations_one_dimensional))
        # It is nescessary to output a tensor of size 10, for 10 different output classes
        result = self.fc3(activations_one_dimensional)
        return result

    # This function is slow because all four function calls for 4 directions are
    # executed sequentially. It isn't entirely clear how to optimize this.
    # See the discussion at:
    # https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360
    def forward_multi_directional_multi_dimensional_rnn(self, x):
        print("list(x.size()): " + str(list(x.size())))

        # Original order
        activations_unskewed_direction_one = self.compute_multi_dimensional_rnn_one_direction(x)
        activations_one_dimensional_one = activations_unskewed_direction_one.view(-1, 1024)

        # Flipping 2nd dimension
        activations_unskewed_direction_two = self.compute_multi_dimensional_rnn_one_direction(
            util.tensor_flipping.flip(x, 2))
        activations_one_dimensional_two = activations_unskewed_direction_two.view(-1, 1024)

        #print("activations_one_dimensional_two: " + str(activations_one_dimensional_two))

        # Flipping 3th dimension
        activations_unskewed_direction_three = self.compute_multi_dimensional_rnn_one_direction(
            util.tensor_flipping.flip(x, 3))
        activations_one_dimensional_three = activations_unskewed_direction_three.view(-1, 1024)

        # Flipping 2nd and 3th dimension combined
        activations_unskewed_direction_four = self.compute_multi_dimensional_rnn_one_direction(
            util.tensor_flipping.flip(util.tensor_flipping.flip(x, 2), 3))
        activations_one_dimensional_four = activations_unskewed_direction_four.view(-1, 1024)

        activations_combined = torch.cat((activations_one_dimensional_one, activations_one_dimensional_two,
                                         activations_one_dimensional_three, activations_one_dimensional_four), 1)

        #print("activations_combined: " + str(activations_combined))

        # print("activations_one_dimensional: " + str(activations_one_dimensional))
        # It is nescessary to output a tensor of size 10, for 10 different output classes
        result = self.fc3(activations_combined)
        return result



    # Input tensor x is a batch of image tensors
    def forward(self, x):
        if self.compute_multi_directional:
            # return self.forward_multi_directional_multi_dimensional_rnn(x)
            return self.forward_multi_directional_multi_dimensional_function_fast(x)
        else:
            return self.forward_one_directional_multi_dimensional_rnn(x)

    def _final_activation_function(self, final_activation_function_input):
        # print("final_activation_function_input: " + str(final_activation_function_input))
        return self.fc3(final_activation_function_input)

    # Needs to be implemented in the subclasses
    def _compute_multi_dimensional_function_one_direction(self, function_input):
        return self.compute_multi_dimensional_rnn_one_direction(function_input)

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
