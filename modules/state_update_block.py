import torch
import torch.nn as nn
from torch.autograd import Variable
from util.utils import Utils
import torch.nn.functional as F
from modules.inside_model_gradient_clipping import InsideModelGradientClamping

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class StateUpdateBlock():
    def __init__(self, hidden_states_size: int):
        self.hidden_states_size = hidden_states_size
        self.state_one_convolution = nn.Conv1d(self.hidden_states_size,
                                                      self.hidden_states_size, 1)
        self.state_two_convolution = nn.Conv1d(self.hidden_states_size,
                                                      self.hidden_states_size, 1)

    # Method takes an input column, and shifts it one position down along the last
    # dimension, adding a row of zeros at the top. The last row is removed, so that
    # the size of the output remains the same size as the input
    @staticmethod
    def get_shifted_column(previous_state_column, hidden_states_size: int):
        previous_memory_state_column_shifted = previous_state_column.clone()
        height = previous_state_column.size(2)
        zeros_padding = Variable(torch.zeros(previous_state_column.size(0), hidden_states_size, 1))
        if Utils.use_cuda():
            zeros_padding = zeros_padding.cuda()
        skip_first_sub_tensor = previous_memory_state_column_shifted[:, :, 0:(height - 1)]
        # print("zeros padding" + str(zeros_padding))
        # print("skip_first_sub_tensor: " + str(skip_first_sub_tensor))
        previous_memory_state_column_shifted = torch. \
            cat((zeros_padding, skip_first_sub_tensor), 2)
        # print("Returning previous_memory_state_column_shifted: " + str(previous_memory_state_column_shifted))
        return previous_memory_state_column_shifted

    # This is a faster implementation of the get_shifted_column method
    # that avoids use of the torch.cat method, but instead uses F.pad
    @staticmethod
    def get_shifted_column_fast(previous_state_column, clamp_gradients: bool):
        #print("previous_state_column: " + str(previous_state_column))
        previous_state_column_4_dim = previous_state_column.unsqueeze(2) # add a fake height

        if clamp_gradients:
            InsideModelGradientClamping.\
                register_gradient_clamping_default_clamping_bound(previous_state_column_4_dim,
                                                                  "get_shifted_column_fast - unsqueeze")

        # See: https://github.com/pytorch/pytorch/issues/1128
        previous_state_column_with_padding = F.pad(previous_state_column_4_dim,
                                                   (1, 0, 0, 0)).view(previous_state_column.size(0),
                                                                      previous_state_column.size(1),
                                                                      -1)[:, :, 0: previous_state_column.size(2)]
        if clamp_gradients:
            InsideModelGradientClamping.\
                register_gradient_clamping_default_clamping_bound(previous_state_column_with_padding,
                                                                  "get_shifted_column_fast - pad")


        # print("previous_state_column_with_padding: " + str(previous_state_column_with_padding))
        result = previous_state_column_with_padding[:, :, 0: previous_state_column.size(2)]
        # print("result: " + str(result))
        return result

    @staticmethod
    def get_previous_state_column(previous_state_column, state_index: int):
        # print("previous memory state column: " + str(previous_memory_state_column))
        if state_index == 2:
            return StateUpdateBlock.get_shifted_column_fast(previous_state_column)
        return previous_state_column

    @staticmethod
    def compute_weighted_state_input_state_one(state_convolution, previous_state_column):
        return state_convolution(previous_state_column)

    # The weighted state input for the second state is computed over the shifted previous
    # state column. This is necessary to get the right predecessor, while using the
    # skewed input trick from the pixel RNN paper as done in this implementation of
    # MultiDimensionalLSTM
    @staticmethod
    def compute_weighted_state_input_state_two(state_convolution, previous_state_column):
        return state_convolution(StateUpdateBlock.get_shifted_column_fast(previous_state_column))

    def compute_weighted_states_input(self, previous_state_column):
        state_one_result = self.state_one_convolution(previous_state_column)
        state_two_result = self.state_two_convolution(StateUpdateBlock.get_shifted_column_fast(previous_state_column))
        result = state_one_result + state_two_result
        return result

    def get_state_convolutions_as_list(self):
        return [self.state_one_convolution, self.state_two_convolution]

    def set_bias_for_convolutions(self, bias_value):
        self.state_one_convolution.bias.data.fill_(bias_value)
        self.state_two_convolution.bias.data.fill_(bias_value)
