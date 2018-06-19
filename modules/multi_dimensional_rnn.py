""" This class will implement Multi-dimensional RNN modules and also Multi-dimensional LSTM modules
    These will be based on a generalization of the native pytorch RNN code
    from:  https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
"""
import math
import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn
from util.image_input_transformer import ImageInputTransformer
import torch.nn as nn
import util.tensor_flipping
from abc import abstractmethod
from modules.state_update_block import StateUpdateBlock
from modules.parallel_multiple_state_weightings_computation import ParallelMultipleStateWeightingsComputation
from modules.size_two_dimensional import SizeTwoDimensional
from util.tensor_chunking import TensorChunking


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


# This Module class stores a MultiDimensionalRNN or MultiDimensionalLSTM
# and uses that to compute the initial activation, then flattens this
# activation and feeds it to a final linear layer to compute the final output
# This increases generality, by avoiding the need to provide the image size to the
# MultiDimensionalRNN / MultiDimensionalLSTM modules, which should be generic and
# working for arbitrary input sizes. An issue is that within a batch at least, all
# inputs should be of the same size. But this can probably be fixed using padding.
class MultiDimensionalRNNToSingleClassNetwork(torch.nn.Module):
    def __init__(self, multi_dimensional_rnn, input_size: SizeTwoDimensional):
        super(MultiDimensionalRNNToSingleClassNetwork, self).__init__()
        self.multi_dimensional_rnn = multi_dimensional_rnn
        self.input_size = input_size
        self.number_of_output_dimensions = multi_dimensional_rnn.get_number_of_output_dimensions(input_size)
        self.fc3 = nn.Linear(self.number_of_output_dimensions, 10)
        # print("self.fc3 : " + str(self.fc3))
        # print("self.fc3.weight: " + str(self.fc3.weight))
        # print("self.fc3.bias: " + str(self.fc3.bias))

    @staticmethod
    def create_multi_dimensional_rnn_to_single_class_network(multi_dimensional_rnn, input_size: SizeTwoDimensional):
        return MultiDimensionalRNNToSingleClassNetwork(multi_dimensional_rnn, input_size)

    def set_training(self, training):
        self.multi_dimensional_rnn.set_training(training)

    def forward(self, x):
        mdrnn_activations = self.multi_dimensional_rnn(x)
        activations_one_dimensional = mdrnn_activations.view(-1, self.number_of_output_dimensions)
        return self.fc3(activations_one_dimensional)


class MultiDimensionalRNNBase(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_states_size: int,
                 compute_multi_directional: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalRNNBase, self).__init__()

        self.input_channels = input_channels
        self.selection_tensors_dictionary = dict([])
        self.nonlinearity = nonlinearity
        self.hidden_states_size = hidden_states_size
        self.compute_multi_directional_flag = compute_multi_directional

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        result = input_size.height * input_size.width \
                 * self.get_hidden_states_size()
        if self.compute_multi_directional():
            result = result * 4
        return result

    # Selection tensors are dynamically stored in a dictionary or retrieved from it
    # if already present. This avoids the need to keep a batch size to have a "default"
    # selection tensor, which reduces the usability of the code for changing batch
    # sizes
    def get_or_add_and_get_selection_tensor(self, x):
        number_of_examples = x.size(0)
        # print("number of examples: " + str(number_of_examples))

        if number_of_examples in self.selection_tensors_dictionary:
            selection_tensor = self.selection_tensors_dictionary[number_of_examples]
        else:
            selection_tensor = TensorChunking.create_torch_indices_selection_tensor(
                number_of_examples, 4)
            if MultiDimensionalRNNBase.use_cuda():
                device = x.get_device()
                selection_tensor = selection_tensor.to(device)
            self.selection_tensors_dictionary[number_of_examples] = selection_tensor
        return selection_tensor

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



        # Original order
        activations_unskewed = self._compute_multi_dimensional_function_one_direction(x_multiple_directions)


        # print("activations_unskewed: " + str(activations_unskewed))
        # print("selection_tensor: " + str(selection_tensor))

        # Using torch.index_select we can bring together the activations of the four
        # different rotations, while avoiding use of a for loop, making the whole thing
        # hopefully faster
        selection_tensor = self.get_or_add_and_get_selection_tensor(x)
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
        result = activations_one_dimensional
        return result

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
        # print("input_column.size(): " + str(input_column.size()))
        # print("state_columns_combined.size(): " + str(state_columns_combined.size()))
        state_plus_input = state_columns_combined + input_column
        # print("input_column: " + str(input_column))
        # print("state_plus_input: " + str(state_plus_input))
        return state_plus_input

    def get_hidden_states_size(self):
        return self.hidden_states_size

    def compute_multi_directional(self):
        return self.compute_multi_directional_flag


class MultiDimensionalRNNAbstract(MultiDimensionalRNNBase):
    def __init__(self, input_channels: int, hidden_states_size, batch_size, compute_multi_directional: bool,
                 use_dropout: bool, training: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalRNNAbstract, self).__init__(input_channels, hidden_states_size, batch_size,
                                                          compute_multi_directional,
                                                          nonlinearity)
        self.input_convolution = nn.Conv2d(self.input_channels,
                                           self.hidden_states_size, 1)
        # For multi-directional rnn
        self.use_dropout = use_dropout
        self.training = training

    # Needs to be implemented in the subclasses
    # This method compute the state update for the two input dimension and
    # returns the summed result
    @abstractmethod
    def _compute_weighted_states_input_summed(self, previous_state_column):
        raise RuntimeError("not implemented")

    def compute_multi_dimensional_rnn_one_direction(self, x):
        # Step 1: Create a skewed version of the input image
        # skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(x)

        # The image is 3-dimensional, but the convolution somehow
        # requires 4-dimensional input. Whdevice=gpus[0])y? This is probably because
        # one extra dimension is for channels, and the fourth dimension is for
        # doing multiple examples in a mini-batch in parallel
        # http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        # http://pytorch.org/docs/master/torch.html#torch.unsqueeze
        skewed_images_variable = ImageInputTransformer.create_skewed_images_variable_four_dim(x)
        image_height = x.size(2)
        number_of_examples = x.size(0)
        # print("image height: " + str(image_height))
        previous_hidden_state_column = torch.zeros(self.input_channels,
                                                   self.hidden_states_size,
                                                   image_height)

        if MultiDimensionalRNNBase.use_cuda():
            device = x.get_device()
            previous_hidden_state_column = previous_hidden_state_column.to(device)

        # print("image_height: " + str(image_height))

        skewed_image_columns = skewed_images_variable.size(3)

        input_matrix = self.input_convolution(skewed_images_variable)
        # print("input_matrix: " + str(input_matrix))

        activations = list([])

        for column_number in range(0, skewed_image_columns):
            # Compute convolution on previous state column vector padded with zeros
            state_columns_combined = self._compute_weighted_states_input_summed(previous_hidden_state_column)

            # print("state_column.size(): " + str(state_column.size()))
            state_plus_input = MultiDimensionalRNNBase.compute_states_plus_input(input_matrix,
                                                                                 column_number,
                                                                                 state_columns_combined)
            activation_column = self.get_activation_function()(state_plus_input)
            #print("activation column: " + str(activation_column))
            previous_hidden_state_column = activation_column
            activations.append(activation_column)
            #print("activations length: " + str(len(activations)))

        original_image_columns = x.size(3)
        print(">>> x.size(): " + str(x.size()))
        skewed_image_rows = skewed_images_variable.size(2)
        #print("Skewed image rows: " + str(skewed_image_rows))

        #print("activations: " + str(activations))
        activations_unskewed = MultiDimensionalRNNBase.\
            extract_unskewed_activations_from_activation_columns(activations, original_image_columns,
                                                                 skewed_image_columns, skewed_image_rows)
        #print("activations_unskewed: " + str(activations_unskewed))
        return activations_unskewed

    def forward_one_directional_multi_dimensional_rnn(self, x):
        activations_unskewed = self.compute_multi_dimensional_rnn_one_direction(x)
        # print("activations_one_dimensional: " + str(activations_one_dimensional))

        return activations_unskewed

    # This function is slow because all four function calls for 4 directions are
    # executed sequentially. It isn't entirely clear how to optimize this.
    # See the discussion at:
    # https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360
    def forward_multi_directional_multi_dimensional_rnn(self, x):
        print("list(x.size()): " + str(list(x.size())))

        # Original order
        activations_unskewed_direction_one = self.compute_multi_dimensional_rnn_one_direction(x)

        # Flipping 2nd dimension
        height_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, False)
        activations_unskewed_direction_two_flipped = self.compute_multi_dimensional_rnn_one_direction(
            height_flipping.flip(x))
        # Flip back the activations to get the retrieve the original orientation
        activations_unskewed_direction_two = height_flipping.flip(activations_unskewed_direction_two_flipped)

        #print("activations_one_dimensional_two: " + str(activations_one_dimensional_two))

        # Flipping 3th dimension
        width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(False, True)
        activations_unskewed_direction_three_flipped = self.compute_multi_dimensional_rnn_one_direction(
            width_flipping.flip(x))
        # Flip back the activations to get the retrieve the original orientation
        activations_unskewed_direction_three = width_flipping.flip(activations_unskewed_direction_three_flipped)

        # Flipping 2nd and 3th dimension combined
        height_and_width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, True)
        activations_unskewed_direction_four_flipped = self.compute_multi_dimensional_rnn_one_direction(
            height_and_width_flipping.flip(x))
        # Flip back the activations to get the retrieve the original orientation
        activations_unskewed_direction_four = height_and_width_flipping.\
            flip(activations_unskewed_direction_four_flipped)

        activations_combined = torch.cat((activations_unskewed_direction_one, activations_unskewed_direction_two,
                                         activations_unskewed_direction_three, activations_unskewed_direction_four), 1)

        #print("activations_combined: " + str(activations_combined))

        # print("activations_one_dimensional: " + str(activations_one_dimensional))
        # It is nescessary to output a tensor of size 10, for 10 different output classes
        result = activations_combined
        return result

    # Input tensor x is a batch of image tensors
    def forward(self, x):
        if self.compute_multi_directional_flag:
            # return self.forward_multi_directional_multi_dimensional_rnn(x)
            return self.forward_multi_directional_multi_dimensional_function_fast(x)
        else:
            return self.forward_one_directional_multi_dimensional_rnn(x)

    def _compute_multi_dimensional_function_one_direction(self, function_input):
        return self.compute_multi_dimensional_rnn_one_direction(function_input)


class MultiDimensionalRNN(MultiDimensionalRNNAbstract):
    def __init__(self, hidden_states_size, compute_multi_directional: bool,
                 use_dropout: bool, training: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalRNN, self).__init__(hidden_states_size,
                                                  compute_multi_directional,
                                                  use_dropout, training,
                                                  nonlinearity)
        self.state_update_block = StateUpdateBlock(hidden_states_size)
        # This is necessary to make sure things are stored on the same gpu, otherwise
        # pytorch doesn't realizes these convolutions are part of this module
        self.state_convolutions = nn.ModuleList(self.state_update_block.get_state_convolutions_as_list())

    @staticmethod
    def create_multi_dimensional_rnn(hidden_states_size: int, compute_multi_directional: bool,
                                     use_dropout: bool,
                                     nonlinearity="tanh"):
        return MultiDimensionalRNN(hidden_states_size, compute_multi_directional,
                                   use_dropout, True,
                                   nonlinearity)

    # This method compute the state update for the two input dimension and
    # returns the summed result
    def _compute_weighted_states_input_summed(self, previous_state_column):
        result = self.state_update_block.compute_weighted_states_input(previous_state_column)

        if self.use_dropout:
            result = F.dropout(result, p=0.2, training=self.training)

        return result


class MultiDimensionalRNNFast(MultiDimensionalRNNAbstract):
    def __init__(self, input_channels: int, hidden_states_size, compute_multi_directional: bool,
                 use_dropout: bool,
                 training: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalRNNFast, self).__init__(input_channels, hidden_states_size,
                                                      compute_multi_directional,
                                                      use_dropout,
                                                      training,
                                                      nonlinearity)
        self.parallel_multiple_state_weighting_computation = \
            ParallelMultipleStateWeightingsComputation.\
            create_parallel_multiple_state_weighting_computation(self.hidden_states_size, 2,
                                                                 use_dropout)

        # This is necessary to make sure things are stored on the same gpu, otherwise
        # pytorch doesn't realizes these convolutions are part of this module
        self.state_convolutions = nn.ModuleList(
            self.parallel_multiple_state_weighting_computation.get_state_convolutions_as_list())

    @staticmethod
    def create_multi_dimensional_rnn_fast(hidden_states_size: int,
                                          compute_multi_directional: bool, use_dropout: bool,
                                          nonlinearity="tanh"):
        return MultiDimensionalRNNFast(hidden_states_size,
                                       compute_multi_directional, use_dropout, True,
                                       nonlinearity)

    # This method compute the state update for the two input dimension and
    # returns the summed result
    def _compute_weighted_states_input_summed(self, previous_state_column):
        return self.parallel_multiple_state_weighting_computation.\
            compute_summed_outputs_every_pair(previous_state_column)[0]

    def set_training(self, training):
        self.parallel_multiple_state_weighting_computation.set_training(training)


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
