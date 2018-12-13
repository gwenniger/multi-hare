from modules.state_update_block import StateUpdateBlock
from modules.parallel_multiple_state_weightings_computation import ParallelMultipleStateWeightingsComputation
from modules.parallel_multiple_state_weightings_computation import \
    ParallelMultipleInputWeightingsComputation
from modules.parallel_multiple_input_convolutions_computation import ParallelMultipleInputConvolutionsComputation
from abc import abstractmethod
import torch.nn as nn
from torch.nn.modules.module import Module
import torch
from util.tensor_utils import TensorUtils


class MultiDimensionalLSTMParametersBase(Module):
    # https://github.com/pytorch/pytorch/issues/750
    FORGET_GATE_BIAS_INIT = 1  # Good for normal (MD)LSTM, but not for Leaky LP cells!

    def __init__(self, hidden_states_size,
                 input_channels: int, use_dropout: bool):
        super(MultiDimensionalLSTMParametersBase, self).__init__()
        self.input_channels = input_channels
        self.hidden_states_size = hidden_states_size
        self.use_dropout = use_dropout

    def set_bias_forget_gates_image_input(self):
        self.forget_gate_one_input_convolution.bias.data.fill_(
            OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        )
        self.forget_gate_two_input_convolution.bias.data.fill_(
            OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        )

    @abstractmethod
    def prepare_input_convolutions(self, skewed_images_variable):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        raise RuntimeError("not implemented")

    # @abstractmethod
    # def get_input_input_matrix(self):
    #     raise RuntimeError("not implemented")

    @abstractmethod
    def get_input_input_column(self, column_index):
        raise RuntimeError("not implemented")


    # @abstractmethod
    # def get_input_gate_input_matrix(self):
    #     raise RuntimeError("not implemented")

    @abstractmethod
    def get_input_gate_input_column(self, column_index):
        raise RuntimeError("not implemented")

    # @abstractmethod
    # def get_forget_gate_one_input_matrix(self):
    #     raise RuntimeError("not implemented")

    @abstractmethod
    def get_forget_gate_one_input_column(self, column_index):
        raise RuntimeError("not implemented")

    # @abstractmethod
    # def get_forget_gate_two_input_matrix(self):
    #     raise RuntimeError("not implemented")

    @abstractmethod
    def get_forget_gate_two_input_column(self, column_index):
        raise RuntimeError("not implemented")

    # @abstractmethod
    # def get_output_gate_input_matrix(self):
    #     raise RuntimeError("not implemented")

    @abstractmethod
    def get_output_gate_input_column(self, column_index):
        raise RuntimeError("not implemented")



    # input_input_matrix = mdlstm_parameters.input_input_convolution(skewed_images_variable)
    # # print("input_input_matrix.size(): " + str(input_input_matrix.size()))
    # input_gate_input_matrix = mdlstm_parameters.input_gate_input_convolution(skewed_images_variable)
    # forget_gate_one_input_matrix = mdlstm_parameters.forget_gate_one_input_convolution(skewed_images_variable)
    # forget_gate_two_input_matrix = mdlstm_parameters.forget_gate_two_input_convolution(skewed_images_variable)
    # output_gate_input_matrix = mdlstm_parameters.output_gate_input_convolution(skewed_images_variable)




    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_input_hidden_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_input_gate_hidden_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_input_gate_memory_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_forget_gate_one_hidden_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_forget_gate_two_hidden_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_forget_gate_one_memory_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_forget_gate_two_memory_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def get_output_gate_hidden_state_column(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def set_bias_forget_gates_to_one(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def set_bias_everything_to_zero(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def set_training(self, training):
        raise RuntimeError("not implemented")

    @abstractmethod
    def compute_output_gate_memory_state_weighted_input(self, previous_memory_state_column):
        raise RuntimeError("not implemented")

    # This class extends Module so as to make sure that the parameters
    # are properly copied (to the right cuda device) when using nn.DataParallel(model)
    # and the to(device) method from  the Module base class
    # http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html
    # The class is not however meant to be used as a stand-alone Module, so forward
    # is not implemented
    def forward(self, x):
        raise NotImplementedError


class MDLSTMParametersPrecomputedInputMatricesBase(MultiDimensionalLSTMParametersBase):
    def __init__(self, hidden_states_size,
                 input_channels: int, use_dropout: bool):
        super(MDLSTMParametersPrecomputedInputMatricesBase, self).__init__(
            hidden_states_size, input_channels, use_dropout)

    @abstractmethod
    def get_input_input_matrix(self):
        raise RuntimeError("not implemented")

    def get_input_input_column(self, column_index):
        return self.get_input_input_matrix()[:, :, :, column_index]

    @abstractmethod
    def get_input_gate_input_matrix(self):
        raise RuntimeError("not implemented")

    def get_input_gate_input_column(self, column_index):
        return self.get_input_gate_input_matrix()[:, :, :, column_index]

    @abstractmethod
    def get_forget_gate_one_input_matrix(self):
        raise RuntimeError("not implemented")

    def get_forget_gate_one_input_column(self, column_index):
        return self.get_forget_gate_one_input_matrix()[:, :, :, column_index]

    @abstractmethod
    def get_forget_gate_two_input_matrix(self):
        raise RuntimeError("not implemented")

    def get_forget_gate_two_input_column(self, column_index):
        return self.get_forget_gate_two_input_matrix()[:, :, :, column_index]

    @abstractmethod
    def get_output_gate_input_matrix(self):
        raise RuntimeError("not implemented")

    def get_output_gate_input_column(self, column_index):
        return self.get_output_gate_input_matrix()[:, :, :, column_index]


class OneDirectionalMultiDimensionalLSTMParametersBase(MDLSTMParametersPrecomputedInputMatricesBase):
    def __init__(self, hidden_states_size,
                 input_channels: int, use_dropout: bool):
        super(OneDirectionalMultiDimensionalLSTMParametersBase, self).__init__(
            hidden_states_size, input_channels, use_dropout)

        # Memory state convolutions
        self.output_gate_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                              self.hidden_states_size, 1)
        nn.init.xavier_uniform_(self.output_gate_memory_state_convolution.weight)

    def compute_output_gate_memory_state_weighted_input(self, previous_memory_state_column):
        return StateUpdateBlock.compute_weighted_state_input_state_one(
            self.output_gate_memory_state_convolution,
            previous_memory_state_column)



class MultiDimensionalLSTMParametersOneDirection(OneDirectionalMultiDimensionalLSTMParametersBase):
    def __init__(self, hidden_states_size, input_channels, use_dropout: bool):
        super(MultiDimensionalLSTMParametersOneDirection, self).__init__(hidden_states_size, input_channels,
                                                                         use_dropout)

        self.input_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.input_gate_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.forget_gate_one_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions()
        self.forget_gate_two_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.output_gate_hidden_state_update_block = StateUpdateBlock(hidden_states_size)

        # Memory state convolutions
        self.input_gate_memory_state_update_block = StateUpdateBlock(hidden_states_size)
        self.forget_gate_one_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                                  self.hidden_states_size, 1)
        self.forget_gate_two_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                                  self.hidden_states_size, 1)
        self.skewed_images_variable = None
        self.previous_hidden_state_column = None
        self.previous_memory_state_column = None

        self.input_input_convolution = nn.Conv2d(self.input_channels,
                                                 self.hidden_states_size, 1)
        self.input_gate_input_convolution = nn.Conv2d(self.input_channels,
                                                      self.hidden_states_size, 1)
        self.forget_gate_one_input_convolution = nn.Conv2d(self.input_channels,
                                                           self.hidden_states_size, 1)
        self.forget_gate_two_input_convolution = nn.Conv2d(self.input_channels,
                                                           self.hidden_states_size, 1)
        self.output_gate_input_convolution = nn.Conv2d(self.input_channels,
                                                       self.hidden_states_size, 1)

        # Initialize the input and memory state convolutions with the
        # Xavier Glorot scheme
        self.initialize_input_convolutions_xavier_glorot()

        self.input_input_matrix = None
        self.input_gate_input_matrix = None
        self.forget_gate_one_input_matrix = None
        self.forget_gate_two_input_matrix = None
        self.output_gate_input_matrix = None

        # TODO: Add dropout to the remaining layers

    def initialize_input_convolutions_xavier_glorot(self):
        # Xavier Glorot weight initialization
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        nn.init.xavier_uniform_(self.input_input_convolution.weight)
        nn.init.xavier_uniform_(self.input_gate_input_convolution.weight)
        nn.init.xavier_uniform_(self.forget_gate_one_input_convolution.weight)
        nn.init.xavier_uniform_(self.forget_gate_two_input_convolution.weight)
        nn.init.xavier_uniform_(self.output_gate_input_convolution.weight)

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction(hidden_states_size, input_channels,
                                                               use_dropout: bool):
        return MultiDimensionalLSTMParametersOneDirection(hidden_states_size, input_channels, use_dropout)

    def prepare_input_convolutions(self, skewed_images_variable):
        self.skewed_images_variable = skewed_images_variable

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        self.previous_hidden_state_column = previous_hidden_state_column
        self.previous_memory_state_column = previous_memory_state_column
        self.input_input_matrix = self.get_input_input_matrix()
        self.input_gate_input_matrix = self.get_input_gate_input_matrix()
        self.forget_gate_one_input_matrix = self.get_forget_gate_one_input_matrix()
        self.forget_gate_two_input_matrix = self.get_forget_gate_two_input_matrix()
        self.output_gate_input_matrix = self.get_output_gate_input_matrix()

    def get_input_input_matrix(self):
        input_input_matrix = self.input_input_convolution(self.skewed_images_variable)
        return input_input_matrix

    def get_input_gate_input_matrix(self):
        input_gate_input_matrix = self.input_gate_input_convolution(self.skewed_images_variable)
        return input_gate_input_matrix

    def get_forget_gate_one_input_matrix(self):
        forget_gate_one_input_matrix = self.forget_gate_one_input_convolution(self.skewed_images_variable)
        return forget_gate_one_input_matrix

    def get_forget_gate_two_input_matrix(self):
        forget_gate_two_input_matrix = self.forget_gate_two_input_convolution(self.skewed_images_variable)
        return forget_gate_two_input_matrix

    def get_output_gate_input_matrix(self):
        output_gate_input_matrix = self.output_gate_input_convolution(self.skewed_images_variable)
        return output_gate_input_matrix

    def get_input_hidden_state_column(self):
        input_hidden_state_column = self.input_hidden_state_update_block. \
            compute_weighted_states_input(self.previous_hidden_state_column)
        return input_hidden_state_column

    def get_input_gate_hidden_state_column(self):
        input_gate_hidden_state_column = self.input_gate_hidden_state_update_block. \
            compute_weighted_states_input(self.previous_hidden_state_column)
        return input_gate_hidden_state_column

    def get_input_gate_memory_state_column(self):
        input_gate_memory_state_column = self.input_gate_memory_state_update_block. \
            compute_weighted_states_input(self.previous_memory_state_column)
        return input_gate_memory_state_column

    def get_forget_gate_one_hidden_state_column(self):
        forget_gate_one_hidden_state_column = \
            self.forget_gate_one_hidden_state_update_block.compute_weighted_states_input(
                self.previous_hidden_state_column)
        return forget_gate_one_hidden_state_column

    def get_forget_gate_two_hidden_state_column(self):
        forget_gate_two_hidden_state_column = \
            self.forget_gate_two_hidden_state_update_block.compute_weighted_states_input(
                self.previous_hidden_state_column)
        return forget_gate_two_hidden_state_column

    def get_forget_gate_one_memory_state_column(self):
        forget_gate_memory_state_column = \
            StateUpdateBlock.compute_weighted_state_input_state_one(self.forget_gate_one_memory_state_convolution,
                                                                    self.previous_memory_state_column)
        return forget_gate_memory_state_column

    def get_forget_gate_two_memory_state_column(self):
        forget_gate_memory_state_column = \
            StateUpdateBlock.compute_weighted_state_input_state_two(self.forget_gate_two_memory_state_convolution,
                                                                    self.previous_memory_state_column)
        return forget_gate_memory_state_column

    def get_output_gate_hidden_state_column(self):
        output_gate_hidden_state_column = \
            self.output_gate_hidden_state_update_block.compute_weighted_states_input(
                self.previous_hidden_state_column)
        return output_gate_hidden_state_column

    def set_bias_forget_gates_to_one(self):
        # self.forget_gate_one_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        self.forget_gate_one_memory_state_convolution.bias.data.fill_(
            OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT)

        self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(
            OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        )

        # self.forget_gate_two_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_two_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        self.forget_gate_two_memory_state_convolution.bias.data.fill_(
            OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT)

        self.forget_gate_two_hidden_state_update_block.set_bias_for_convolutions(
            OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        )

        # self.set_bias_forget_gates_image_input()

        # Needs to be implemented in the subclasses

    def set_bias_everything_to_zero(self):
        raise RuntimeError("not implemented")

    def set_training(self, training):

        # TODO: implement this
        return

    def forward(self, x):
        raise NotImplementedError


    # This implementation of MultiDimensionalLSTMParametersOneDirectionBase uses special 1d convolutions wrapped by the
# ParallelMultipleStateWeightingsComputation to perform several (N) 1d convolutions over the same input together, using
# a single convolution with N times as many outputs
# https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360
class MultiDimensionalLSTMParametersOneDirectionFast(OneDirectionalMultiDimensionalLSTMParametersBase):
    def __init__(self, hidden_states_size, input_channels, clamp_gradients: bool,
                 use_dropout: bool):
        super(MultiDimensionalLSTMParametersOneDirectionFast, self).__init__(hidden_states_size, input_channels,
                                                                             use_dropout)

        self.parallel_multiple_input_convolutions_computation = ParallelMultipleInputConvolutionsComputation.\
            create_parallel_multiple_input_convolutions_computation(self.input_channels,
                                                                    self.hidden_states_size,
                                                                    5,
                                                                    clamp_gradients,
                                                                    use_dropout)

        # There are five paired input weightings, in this case, pairs of previous hidden state
        # inputs, namely for: 1) the input, 2) the input gate, 3) forget gate one,
        # 4) forget gate two, 5) the output gate
        self.parallel_hidden_state_column_computation = ParallelMultipleStateWeightingsComputation.\
            create_parallel_multiple_state_weighting_computation(hidden_states_size, 5, clamp_gradients, use_dropout)

        self.parallel_memory_state_column_computation = ParallelMultipleStateWeightingsComputation.\
            create_parallel_multiple_state_weighting_computation(hidden_states_size, 2, clamp_gradients, use_dropout)

        self.input_matrices = None
        self.node_hidden_state_columns = None
        self.node_memory_state_columns = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction_fast(hidden_states_size, input_channels,
                                                                    clamp_gradients: bool,
                                                                    use_dropout: bool):
        return MultiDimensionalLSTMParametersOneDirectionFast(hidden_states_size, input_channels,
                                                              clamp_gradients, use_dropout)

    def reset_next_input_column_index(self):
        # Implemented for compatibility with multi_dimensional_lst class
        return

    def prepare_input_convolutions(self, skewed_images_variable):
        self.input_matrices = self.parallel_multiple_input_convolutions_computation.\
            compute_result_and_split_into_output_elements(skewed_images_variable)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column,  mask: torch.Tensor):
        # The hidden state columns for the different computational nodes:
        # 1) the input, 2) the input gate, 3) forget gate one,
        # 4) forget gate two, 5) the output gate
        self.node_hidden_state_columns = \
            self.parallel_hidden_state_column_computation.\
                compute_summed_outputs_every_pair(previous_hidden_state_column, mask)
        # print("self.node_hidden_state_columns: " + str(self.node_hidden_state_columns))
        self.previous_memory_state_column = previous_memory_state_column

        self.node_memory_state_columns = self.\
            parallel_memory_state_column_computation.\
            compute_result_and_split_into_pairs_with_second_pair_element_shifted(previous_memory_state_column, mask)

    def get_input_input_matrix(self):
        return self.input_matrices[0]

    def get_input_gate_input_matrix(self):
        return self.input_matrices[1]

    def get_forget_gate_one_input_matrix(self):
        return self.input_matrices[2]

    def get_forget_gate_two_input_matrix(self):
        return self.input_matrices[3]

    def get_output_gate_input_matrix(self):
        return self.input_matrices[4]

    def get_input_hidden_state_column(self):
        return self.node_hidden_state_columns[0]

    def get_input_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[1]

    def get_forget_gate_one_hidden_state_column(self):
        return self.node_hidden_state_columns[2]

    def get_forget_gate_two_hidden_state_column(self):
        return self.node_hidden_state_columns[3]

    def get_output_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[4]

    def get_input_gate_memory_state_column(self):
        input_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[0]
        input_gate_memory_state_column = input_gate_memory_state_column_part_pair[0] + \
            input_gate_memory_state_column_part_pair[1]
        return input_gate_memory_state_column

    def get_forget_gate_one_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[0]
        return forget_gate_memory_state_column

    def get_forget_gate_two_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[1]
        return forget_gate_memory_state_column

    def set_bias_forget_gates_memory_states_input(self):
        # self.forget_gate_one_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_memory_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # print("before: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #        str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))
        start_index = int(self.parallel_memory_state_column_computation.parallel_convolution.bias.data.size(0) / 2)
        end_index = self.parallel_memory_state_column_computation.parallel_convolution.bias.data.size(0)
        # print("start index: " + str(start_index) + " end index: " + str(end_index))
        for index in range(start_index, end_index):
            self.parallel_memory_state_column_computation.parallel_convolution.bias.data[index] = \
                OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        # print("after: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #      str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_hidden_states_input(self):
        start_index = self.hidden_states_size * 2 * 2
        end_index = self.hidden_states_size * 2 * 4
        # print("start index: " + str(start_index) + " end index: " + str(end_index))
        for index in range(start_index, end_index):
            self.parallel_hidden_state_column_computation.parallel_convolution.bias.data[index] = \
                OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        # print("after: self.parallel_hidden_state_column_computation.parallel_convolution.bias.data: " +
        #      str(self.parallel_hidden_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_to_one(self):
        # self.set_bias_forget_gates_image_input()
        self.set_bias_forget_gates_memory_states_input()
        self.set_bias_forget_gates_hidden_states_input()

        # Needs to be implemented in the subclasses

    def set_bias_everything_to_zero(self):
        # Set bias to zero
        with torch.no_grad():
            self.parallel_hidden_state_column_computation.parallel_convolution.bias.zero_()
            self.parallel_memory_state_column_computation.parallel_convolution.bias.zero_()
            self.parallel_multiple_input_convolutions_computation.parallel_convolution.bias.zero_()

    def set_training(self, training):
        self.parallel_hidden_state_column_computation.set_training(training)
        self.parallel_memory_state_column_computation.set_training(training)

    def forward(self, x):
        raise NotImplementedError


# This implementation of MultiDimensionalLSTMParametersOneDirectionBase uses special 1d convolutions wrapped by the
# ParallelMultipleStateWeightingsComputation to perform several (N) 1d convolutions over the same input together, using
# a single convolution with N times as many outputs
# https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360
class MultiDimensionalLSTMParametersOneDirectionFullyParallel(OneDirectionalMultiDimensionalLSTMParametersBase):
    def __init__(self, hidden_states_size, input_channels, clamp_gradients: bool,
                 use_dropout: bool):
        super(MultiDimensionalLSTMParametersOneDirectionFullyParallel, self).__init__(hidden_states_size, input_channels,
                                                                             use_dropout)

        self.parallel_multiple_input_convolutions_computation = ParallelMultipleInputConvolutionsComputation.\
            create_parallel_multiple_input_convolutions_computation(self.input_channels,
                                                                    self.hidden_states_size,
                                                                    5,
                                                                    clamp_gradients,
                                                                    use_dropout)

        # There are five paired input weightings for the previous hidden state:
        #  in this case, pairs of previous hidden state
        # inputs, namely for: 1) the input, 2) the input gate, 3) forget gate one,
        # 4) forget gate two, 5) the output gate
        # There are also 2 paired input weightings for the previous memory state

        number_of_paired_input_weightings_per_group = list([5, 2])
        self.parallel_hidden_and_memory_state_column_computation = \
            ParallelMultipleStateWeightingsComputation.\
            create_parallel_multiple_state_weighting_computation_multiple_groups(
                    hidden_states_size, number_of_paired_input_weightings_per_group,
                    clamp_gradients, use_dropout)

        # self.parallel_hidden_state_column_computation = ParallelMultipleStateWeightingsComputation.\
        #     create_parallel_multiple_state_weighting_computation(hidden_states_size, 5, clamp_gradients, use_dropout)
        #
        # self.parallel_memory_state_column_computation = ParallelMultipleStateWeightingsComputation.\
        #     create_parallel_multiple_state_weighting_computation(hidden_states_size, 2, clamp_gradients, use_dropout)

        self.input_matrices = None
        self.node_hidden_state_columns = None
        self.node_memory_state_columns = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction_fully_parallel(hidden_states_size, input_channels,
                                                                              clamp_gradients: bool,
                                                                              use_dropout: bool):
        return MultiDimensionalLSTMParametersOneDirectionFullyParallel(hidden_states_size, input_channels,
                                                                       clamp_gradients, use_dropout)

    def prepare_input_convolutions(self, skewed_images_variable):
        self.input_matrices = self.parallel_multiple_input_convolutions_computation.\
            compute_result_and_split_into_output_elements(skewed_images_variable)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column,  mask: torch.Tensor):

        node_hidden_and_memory_state_columns = \
            self.parallel_hidden_and_memory_state_column_computation.\
            compute_result_and_split_into_pairs_with_second_pair_element_shifted_multiple_groups(
                list([previous_hidden_state_column, previous_memory_state_column]), mask)

        # Sanity check that the number of output pairs is as expected
        if len(node_hidden_and_memory_state_columns) != 7:
            raise RuntimeError("Error: expected 7 output pairs")

        self.node_hidden_state_columns = ParallelMultipleStateWeightingsComputation.\
            compute_summed_outputs_every_pair_static(node_hidden_and_memory_state_columns[0:5])

        self.previous_memory_state_column = previous_memory_state_column

        self.node_memory_state_columns = node_hidden_and_memory_state_columns[5:7]

    def get_input_input_matrix(self):
        return self.input_matrices[0]

    def get_input_gate_input_matrix(self):
        return self.input_matrices[1]

    def get_forget_gate_one_input_matrix(self):
        return self.input_matrices[2]

    def get_forget_gate_two_input_matrix(self):
        return self.input_matrices[3]

    def get_output_gate_input_matrix(self):
        return self.input_matrices[4]

    def get_input_hidden_state_column(self):
        return self.node_hidden_state_columns[0]

    def get_input_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[1]

    def get_forget_gate_one_hidden_state_column(self):
        return self.node_hidden_state_columns[2]

    def get_forget_gate_two_hidden_state_column(self):
        return self.node_hidden_state_columns[3]

    def get_output_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[4]

    def get_input_gate_memory_state_column(self):
        input_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[0]
        input_gate_memory_state_column = input_gate_memory_state_column_part_pair[0] + \
            input_gate_memory_state_column_part_pair[1]
        return input_gate_memory_state_column

    def get_forget_gate_one_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[0]
        return forget_gate_memory_state_column

    def get_forget_gate_two_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[1]
        return forget_gate_memory_state_column

    def set_bias_forget_gates_memory_states_input(self):
        # self.forget_gate_one_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_memory_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # print("before: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #        str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))
        start_index = self.hidden_states_size * 2 * 6
        end_index = self.hidden_states_size * 2 * MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            num_paired_hidden_and_memory_state_weightings()
        # print("start index: " + str(start_index) + " end index: " + str(end_index))
        for index in range(start_index, end_index):
            self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.bias.data[index] = \
                OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        # print("after: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #      str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_hidden_states_input(self):
        start_index = self.hidden_states_size * 2 * 2
        end_index = self.hidden_states_size * 2 * 4
        # print("start index: " + str(start_index) + " end index: " + str(end_index))
        for index in range(start_index, end_index):
            self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.bias.data[index] = \
                OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
        # print("after: self.parallel_hidden_state_column_computation.parallel_convolution.bias.data: " +
        #      str(self.parallel_hidden_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_to_one(self):
        # self.set_bias_forget_gates_image_input()
        self.set_bias_forget_gates_memory_states_input()
        self.set_bias_forget_gates_hidden_states_input()

        # Needs to be implemented in the subclasses

    def set_bias_everything_to_zero(self):
        raise RuntimeError("not implemented")

    def set_training(self, training):
        self.parallel_hidden_and_memory_state_column_computation.set_training(training)

    def forward(self, x):
        raise NotImplementedError


class MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution(
     MultiDimensionalLSTMParametersBase):
    NUMBER_OF_INPUT_CONVOLUTIONS_PER_DIRECTION = 5
    NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS = 5
    NUMBER_OF_PAIRED_MEMORY_STATE_WEIGHTINGS = 2

    def __init__(self, hidden_states_size, input_channels,
                 use_dropout: bool, number_of_directions: int,
                 parallel_multiple_input_convolutions_computations,
                 parallel_hidden_and_memory_state_column_computation

                 ):
        super(MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution, self).\
            __init__(hidden_states_size, input_channels, use_dropout)
        self.number_of_directions = number_of_directions

        self.parallel_multiple_input_convolutions_computations = parallel_multiple_input_convolutions_computations

        self.parallel_hidden_and_memory_state_column_computation = parallel_hidden_and_memory_state_column_computation

        # Memory state convolutions
        self.output_gate_memory_state_convolution = nn.Conv1d(self.hidden_states_size * number_of_directions,
                                                              self.hidden_states_size * number_of_directions, 1,
                                                              groups=number_of_directions)
        nn.init.xavier_uniform_(self.output_gate_memory_state_convolution.weight)

        self.input_columns = None
        self.input_column_lists = None
        self.node_hidden_state_columns = None
        self.node_memory_state_columns = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_directional_multi_dimensional_lstm_parameters_parallel_with_separate_input_convolution(
            hidden_states_size, input_channels, clamp_gradients: bool, use_dropout: bool, number_of_directions: int):
        print(">>>create_multi_directional_multi_dimensional_lstm_parameters_fully_parallel...")

        parallel_multiple_input_convolutions_computations = \
            MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            create_parallel_multiple_input_convolution_computations(input_channels, hidden_states_size,
                                                                    clamp_gradients, use_dropout, number_of_directions)

        parallel_hidden_and_memory_state_column_computation = \
            MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            create_parallel_hidden_and_memory_state_column_computation(hidden_states_size, clamp_gradients,
                                                                       use_dropout, number_of_directions)

        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution(
            hidden_states_size, input_channels, use_dropout, number_of_directions,
            parallel_multiple_input_convolutions_computations,
            parallel_hidden_and_memory_state_column_computation)

    @staticmethod
    def create_parallel_hidden_and_memory_state_column_computation(
            hidden_states_size: int, clamp_gradients: bool,
            use_dropout: bool, number_of_directions: int):
        number_of_paired_input_weightings_per_group = list([])
        for i in range(0, number_of_directions):
            number_of_paired_input_weightings_per_group.extend(
                list([MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                     NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS,
                      MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                      NUMBER_OF_PAIRED_MEMORY_STATE_WEIGHTINGS]))

        parallel_hidden_and_memory_state_column_computation = \
            ParallelMultipleStateWeightingsComputation. \
            create_parallel_multiple_state_weighting_computation_multiple_groups(
                hidden_states_size, number_of_paired_input_weightings_per_group,
                clamp_gradients, use_dropout)
        return parallel_hidden_and_memory_state_column_computation

    @staticmethod
    def create_parallel_multiple_input_convolution_computations(
            input_channels: int, hidden_states_size: int,
            clamp_gradients: bool, use_dropout: bool, number_of_directions: int):

        parallel_multiple_input_convolutions_computations = nn.ModuleList([])

        for i in range(0, number_of_directions):
            parallel_multiple_input_convolution = ParallelMultipleInputConvolutionsComputation. \
                create_parallel_multiple_input_convolutions_computation(
                    input_channels, hidden_states_size, 5, clamp_gradients, use_dropout)
            parallel_multiple_input_convolutions_computations.append(parallel_multiple_input_convolution)

        return parallel_multiple_input_convolutions_computations

    """
    This method extracts a list of lists of input columns from input matrices.
    The motivation is that by storing lists of columns, and "consuming" these 
    columns during the column-by column MDLSTM computation, removing them from th 
    lists using the list "pop" method, the memory can be freed up as soon as possible.
    Otherwise the entire matrices must be kept in memory for ten entire computation, 
    when they are no longer required.
    """
    @staticmethod
    def extract_input_column_list_of_lists_from_input_matrices(input_matrices: list):
        input_column_lists = list([])
        for i in range(0, len(input_matrices)):
            column_list_for_matrix = list([])
            input_matrix = input_matrices[i]
            # print(">>> input_matrix.size(): " + str(input_matrix.size()))
            # Add elements in inverted order so that later on the "pop" method
            # can be used to obtain methods in front to back order
            for j in range(input_matrix.size(3) - 1, -1, -1):
                # print("j: " + str(j))
                column = input_matrix[:, :, :, j]
                column_list_for_matrix.append(column)

            input_column_lists.append(column_list_for_matrix)

        return input_column_lists

    def prepare_input_convolutions(self, skewed_images_variable):
        # print("Entered MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution."
        #       "prepare_input_convolutions...")

        if TensorUtils.number_of_dimensions(skewed_images_variable) != 4:
            raise RuntimeError("Error: prepare_input_convolution requires 4 dimensional input")

        # if skewed_images_variable.size(0) != self.number_of_directions:
        #     raise RuntimeError("Error: the size of the first dimension should match the number of directions")


        # print("MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.prepare_input_convolutions - split")
        # print("skewed_images_variable.size(): " + str(skewed_images_variable.size()))
        # First split the image tensor to get the images for the different directions
        skewed_images_variable_list = torch.chunk(skewed_images_variable, 4, 1)

        input_matrices_lists = list([])
        # Then compute the input matrix for each direction

        for i, skewed_image in enumerate(skewed_images_variable_list):
            # print("compute_result_and_split_into_output_elements - direction - " + str(i))
            # print("skewed_image.size(): " + str(skewed_image.size()))
            input_matrices = self.parallel_multiple_input_convolutions_computations[i].\
                            compute_result_and_split_into_output_elements(skewed_image)
            # print(">>> len(input_matrices) for direction: " + str(len(input_matrices)))
            # print("compute_result_and_split_into_output_elements - direction - " + str(i) + " - finished")
            input_matrices_lists.append(input_matrices)

        # # print("MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.prepare_input_convolutions - split")
        # # print("skewed_images_variable.size(): " + str(skewed_images_variable.size()))
        # # First split the image tensor to get the images for the different directions
        # skewed_images_variable_list = torch.split(skewed_images_variable, 1, 0)
        # # Concatenate the four images on the channels direction
        # skewed_image_four_directions = torch.cat(skewed_images_variable_list, 1)

        # input_matrices_lists = self.parallel_multiple_input_convolutions_computations.\
        #     compute_result_and_split_into_output_elements(skewed_images_variable)
        #
        input_matrices_lists_grouped_by_index_concatenated = \
            MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            concatenate_elements_list_of_lists_along_dimension(input_matrices_lists, 1)

        # print(">>> len(input_matrices_lists_grouped_by_index_concatenated): " +
        #       str(len(input_matrices_lists_grouped_by_index_concatenated)))

        self.input_column_lists = \
            MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            extract_input_column_list_of_lists_from_input_matrices(
                    input_matrices_lists_grouped_by_index_concatenated)

        # self.input_column_lists = input_matrices_lists_grouped_by_index_concatenated

        if len(self.input_column_lists) != \
                MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
                        NUMBER_OF_INPUT_CONVOLUTIONS_PER_DIRECTION:
            raise RuntimeError("Error: length of input_matrices after merging matrices for multiple" +
                               " directions should still be " +
                               str(MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                   NUMBER_OF_INPUT_CONVOLUTIONS_PER_DIRECTION)
                               )


    """
    This method takes a list of lists of tensors, and pairs the elements with the same 
    index in the inner lists, concatenating them along the specified dimension.
    Input: N lists of lists of M tensors
    Output: one list L_out of M tensors, with tensor t_i formed by concatenating the 
    tensors t_i_n, with n in [1 ... N]  
    # https://stackoverflow.com/questions/4112265/how-to-zip-lists-in-a-list
    """
    @staticmethod
    def concatenate_elements_list_of_lists_along_dimension(list_of_tensor_lists, dim: int):
        list_of_tensor_lists_lists_re_grouped_by_index = zip(*list_of_tensor_lists)

        list_of_elements_same_index_concateneated = list([])
        for elements in list_of_tensor_lists_lists_re_grouped_by_index:
            concatenated_tensors = torch.cat(elements, dim)
            list_of_elements_same_index_concateneated.append(
                concatenated_tensors)

        return list_of_elements_same_index_concateneated

    @staticmethod
    def concatenate_elements_list_of_lists_of_tuples_along_dimension(list_of_tensor_tuples_lists, dim: int):
        list_of_tensor_tuples_lists_re_grouped_by_index = zip(*list_of_tensor_tuples_lists)

        list_of_elements_same_index_concateneated = list([])
        for elements in list_of_tensor_tuples_lists_re_grouped_by_index:
            tuple_cat_list = list([])
            for i, tuple_index_grouped in enumerate(zip(*elements)):
                # print("index - " + str(i) + " - len(tuple_index_grouped): " + str(len(tuple_index_grouped)))

                concatenated_tensors = torch.cat(tuple_index_grouped, dim)
                # print("concatenated_tensors.size() :" + str(concatenated_tensors.size()))
                tuple_cat_list.append(concatenated_tensors)
            list_of_elements_same_index_concateneated.append(tuple(tuple_cat_list))

        return list_of_elements_same_index_concateneated

    @staticmethod
    def num_paired_hidden_and_memory_state_weightings():
        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
                   NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS + \
                   MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
                   NUMBER_OF_PAIRED_MEMORY_STATE_WEIGHTINGS

    """
    Get the next input columns, removing them from the input columns list
    (So that the memory for them can be freed when they are no longer used)
    """
    def get_next_input_columns(self):
        input_columns = list([])
        for input_columns_list in self.input_column_lists:
            input_columns.append(input_columns_list.pop())
        return input_columns

    def get_next_node_hidden_state_columns(self, node_hidden_and_memory_state_columns):

        node_hidden_state_columns_lists = list([])
        for i in range(0, self.number_of_directions):
            offset = MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
                         num_paired_hidden_and_memory_state_weightings() * i
            node_hidden_state_columns = ParallelMultipleStateWeightingsComputation. \
                compute_summed_outputs_every_pair_static(
                    node_hidden_and_memory_state_columns[offset:offset +
                                                                MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                                         NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS])
            node_hidden_state_columns_lists.append(node_hidden_state_columns)

        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            concatenate_elements_list_of_lists_along_dimension(node_hidden_state_columns_lists, 1)

    def get_next_node_memory_state_columns(self, node_hidden_and_memory_state_columns):

        node_memory_state_columns_lists = list([])
        for i in range(0, self.number_of_directions):
            offset = MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
                         num_paired_hidden_and_memory_state_weightings() * i
            node_hidden_state_columns = \
                node_hidden_and_memory_state_columns[offset +
                                                     MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                                     NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS
                                                     :offset +
                                                      MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                                     num_paired_hidden_and_memory_state_weightings()
                                                     ]
            node_memory_state_columns_lists.append(node_hidden_state_columns)

        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution. \
            concatenate_elements_list_of_lists_of_tuples_along_dimension(node_memory_state_columns_lists, 1)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column,  mask: torch.Tensor):
        # print("Entered MultiDirectionalMultiDimensionalLSTMParametersFullyParallel." +
        #       "prepare_computation_next_column_functions...")

        if previous_hidden_state_column.size(1) != self.hidden_states_size * self.number_of_directions:
            raise RuntimeError("Error: the size of the second dimension of" +
                               " previous_hidden_state_column should match the number of directions" +
                               "times the number of hidden states")

        if previous_memory_state_column.size(1) != self.hidden_states_size * self.number_of_directions:
            raise RuntimeError("Error: the size of the second dimension of" +
                               " memory_state_column should match the number of directions " +
                               "times the number of hidden states")

        previous_hidden_state_columns_split_by_direction = torch.chunk(previous_hidden_state_column,
                                                                       self.number_of_directions, 1)
        previous_memory_state_columns_split_by_direction = torch.chunk(previous_memory_state_column,
                                                                       self.number_of_directions, 1)

        computation_arguments_list = list([])
        for i in range(0, self.number_of_directions):
            computation_arguments_list.append(previous_hidden_state_columns_split_by_direction[i])
            computation_arguments_list.append(previous_memory_state_columns_split_by_direction[i])

        node_hidden_and_memory_state_columns = \
            self.parallel_hidden_and_memory_state_column_computation.\
            compute_result_and_split_into_pairs_with_second_pair_element_shifted_multiple_groups(
                computation_arguments_list, mask)

        # Sanity check that the number of output pairs is as expected
        if len(node_hidden_and_memory_state_columns) != (MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                                         num_paired_hidden_and_memory_state_weightings() *
                                                         self.number_of_directions):
            raise RuntimeError("Error: there are " + str(self.number_of_directions) + " directions, " +
                               "therefore expected " + MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                               num_paired_hidden_and_memory_state_weightings() + " * " + str(self.number_of_directions) +
                               " output pairs")

        self.input_columns = self.get_next_input_columns()
        self.node_hidden_state_columns = self.get_next_node_hidden_state_columns(node_hidden_and_memory_state_columns)
        self.node_memory_state_columns = self.get_next_node_memory_state_columns(node_hidden_and_memory_state_columns)

        self.previous_memory_state_column = previous_memory_state_column

        # print("finished prepare_computation_next_column_functions")

    def compute_output_gate_memory_state_weighted_input(self, previous_memory_state_column):
        if TensorUtils.number_of_dimensions(previous_memory_state_column) != 3:
            raise RuntimeError("Error: prepare_input_convolution requires 3 dimensional input"
                               + " got size: " + str(previous_memory_state_column.size()))

        if previous_memory_state_column.size(1) != self.hidden_states_size * self.number_of_directions:
            raise RuntimeError("Error: the size of the first dimension of" +
                               " memory_state_column should match the number of directions" +
                               " times the number of hidden states")

        # previous_memory_state_columns_split_by_direction = torch.split(previous_memory_state_column, 1, 0)
        # previous_memory_state_column_catted_on_channel_dimension = \
        #     torch.cat(previous_memory_state_columns_split_by_direction, 1)
        #
        # result_catted_on_channel_dimension = StateUpdateBlock.compute_weighted_state_input_state_one(
        #     self.output_gate_memory_state_convolution,
        #     previous_memory_state_column_catted_on_channel_dimension)
        # # print("result_catted_on_channel_dimension.size(): " + str(result_catted_on_channel_dimension.size()))
        # result_split_into_directions = torch.chunk(result_catted_on_channel_dimension, self.number_of_directions, 1)
        # # Re-concatenate the direction results on the batch dimension
        # result = torch.cat(result_split_into_directions, 0)
        # # print("result.size(): " + str(result.size()))
        # return result

        result_catted_on_channel_dimension = StateUpdateBlock.compute_weighted_state_input_state_one(
            self.output_gate_memory_state_convolution,
            previous_memory_state_column)

        return result_catted_on_channel_dimension

    # def get_input_input_matrix(self):
    #     return self.input_column_lists[0]
    #
    # def get_input_gate_input_matrix(self):
    #     return self.input_column_lists[1]
    #
    # def get_forget_gate_one_input_matrix(self):
    #     return self.input_column_lists[2]
    #
    # def get_forget_gate_two_input_matrix(self):
    #     return self.input_column_lists[3]
    #
    # def get_output_gate_input_matrix(self):
    #     return self.input_column_lists[4]
    #
    def get_input_input_column(self, column_index):
        return self.input_columns[0]

    def get_input_gate_input_column(self, column_index):
        return self.input_columns[1]

    def get_forget_gate_one_input_column(self, column_index):
        return self.input_columns[2]

    def get_forget_gate_two_input_column(self, column_index):
        return self.input_columns[3]

    def get_output_gate_input_column(self, column_index):
        return self.input_columns[4]

    def get_input_hidden_state_column(self):
        return self.node_hidden_state_columns[0]

    def get_input_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[1]

    def get_forget_gate_one_hidden_state_column(self):
        return self.node_hidden_state_columns[2]

    def get_forget_gate_two_hidden_state_column(self):
        return self.node_hidden_state_columns[3]

    def get_output_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[4]

    def get_input_gate_memory_state_column(self):
        input_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[0]
        input_gate_memory_state_column = input_gate_memory_state_column_part_pair[0] + \
            input_gate_memory_state_column_part_pair[1]
        return input_gate_memory_state_column

    def get_forget_gate_one_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[0]
        return forget_gate_memory_state_column

    def get_forget_gate_two_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[1]
        return forget_gate_memory_state_column

    def number_of_hidden_and_memory_state_weights_per_direction(self):
        return self.hidden_states_size * 2 * \
               MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
               num_paired_hidden_and_memory_state_weightings()

    def number_of_input_weighting_weights_per_direction(self):
        return self.hidden_states_size * \
               MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
               NUMBER_OF_INPUT_CONVOLUTIONS_PER_DIRECTION

    def set_bias_forget_gates_hidden_states_input(self):
        relative_start_index = self.hidden_states_size * 2 * 2
        relative_end_index = self.hidden_states_size * 2 * 4
        # print("start index: " + str(relative_start_index) + " end index: " + str(relative_end_index))

        for direction_index in range(0, self.number_of_directions):
            offset = self.number_of_hidden_and_memory_state_weights_per_direction() * direction_index

            for index in range(offset + relative_start_index, offset + relative_end_index):
                self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.bias.data[index] = \
                    OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
            # print("after: self.parallel_hidden_state_column_computation.parallel_convolution.bias.data: " +
            #      str(self.parallel_hidden_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_memory_states_input(self):
        # self.forget_gate_one_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_memory_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # print("before: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #        str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))
        relative_start_index = self.hidden_states_size * 2 * 6
        relative_end_index = self.number_of_hidden_and_memory_state_weights_per_direction()
        # print("start index: " + str(relative_start_index) + " end index: " + str(relative_end_index))

        for direction_index in range(0, self.number_of_directions):
            offset = self.number_of_hidden_and_memory_state_weights_per_direction() * direction_index

            for index in range(offset + relative_start_index, offset + relative_end_index):
                self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.bias.data[index] = \
                    OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
            # print("after: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
            #      str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_to_one(self):
        # self.set_bias_forget_gates_image_input()
        self.set_bias_forget_gates_memory_states_input()
        self.set_bias_forget_gates_hidden_states_input()

    def set_bias_everything_to_zero(self):
        raise RuntimeError("not implemented")

    def set_training(self, training):
        self.parallel_hidden_and_memory_state_column_computation.set_training(training)

    def forward(self, x):
        raise NotImplementedError

    def copy_weights_parallel_hidden_and_memory_states_convolution_to_one_directional_mdlstm_parameters(
            self, mdlstm_parameters_one_direction, direction_index):

        relative_start_index = 0
        relative_end_index = self.number_of_hidden_and_memory_state_weights_per_direction()
        for one_directional_mdlstm_index in range(relative_start_index, relative_end_index):
            multi_directional_mdlstm_index = \
                one_directional_mdlstm_index + \
                self.number_of_hidden_and_memory_state_weights_per_direction() * direction_index
            mdlstm_parameters_one_direction.parallel_hidden_and_memory_state_column_computation.\
                parallel_convolution.bias.data[one_directional_mdlstm_index] =\
                self.parallel_hidden_and_memory_state_column_computation.\
                parallel_convolution.bias.data[multi_directional_mdlstm_index]
            mdlstm_parameters_one_direction.parallel_hidden_and_memory_state_column_computation.\
                parallel_convolution.weight.data[one_directional_mdlstm_index, :, :] = \
                self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.\
                weight.data[multi_directional_mdlstm_index, :, :]

    def copy_parallel_multiple_input_convolutions_computation_to_one_directional_mdlstm_parameters(
            self, mdlstm_parameters_one_direction, direction_index):

        mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation = \
                        self.parallel_multiple_input_convolutions_computations[direction_index]

        # relative_start_index = 0
        # relative_end_index = self.number_of_input_weighting_weights_per_direction()
        # for one_directional_mdlstm_index in range(relative_start_index, relative_end_index):
        #     multi_directional_mdlstm_index = \
        #         one_directional_mdlstm_index + \
        #         self.number_of_input_weighting_weights_per_direction() * direction_index
        #     mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
        #         parallel_convolution.bias.data[one_directional_mdlstm_index] = \
        #         self.parallel_multiple_input_convolutions_computations.\
        #             parallel_convolution.bias.data[multi_directional_mdlstm_index]
        #     mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
        #         parallel_convolution.weight.data[one_directional_mdlstm_index, :, :] = \
        #         self.parallel_multiple_input_convolutions_computations.parallel_convolution. \
        #             weight.data[multi_directional_mdlstm_index, :, :]

    def copy_output_gate_memory_state_convolution_to_one_directional_mdlstm_parameters(
            self, mdlstm_parameters_one_direction, direction_index):
        relative_start_index = 0
        relative_end_index = self.hidden_states_size
        for one_directional_mdlstm_index in range(relative_start_index, relative_end_index):
            multi_directional_mdlstm_index = \
                self.hidden_states_size * direction_index + one_directional_mdlstm_index

            mdlstm_parameters_one_direction.output_gate_memory_state_convolution.bias[one_directional_mdlstm_index] = \
                self.output_gate_memory_state_convolution.bias[multi_directional_mdlstm_index]

            mdlstm_parameters_one_direction.output_gate_memory_state_convolution.\
                weight[one_directional_mdlstm_index, :, :] = \
                self.output_gate_memory_state_convolution.weight[multi_directional_mdlstm_index, :, :]

    """
    This methods extracts/creates  a list of one-directional MDLSTM parameters based on the
    current weights of the multi-directional MDLSTM parameters. This is mainly useful for 
    testing purposes. Creating (one-directional) MDLSTMs from the thus extracted parameters, 
    it can be tested whether separately executed one-directional MDLSTMs produce the same results
    as the corresponding outputs for that direction of the multi-directional MDLSTM from which 
    the parameters were taken. This kind of testing is useful, since it is very hard to test 
    directly whether the multi-directional MDLSTM "works". This is actually a general problem 
    for testing neural networks: finding and fixing problems with mismatching tensor sizes etc
    is easy enough. But once something runs without errors, it can be very hard to determine 
    whether it is actually computing the right thing. 
    """
    def create_one_directional_mdlstm_parameters_each_direction_using_current_weights(self):
        result = list([])

        for i in range(0, self.number_of_directions):
            mdlstm_parameters_one_direction = MultiDimensionalLSTMParametersOneDirectionFullyParallel.\
                create_multi_dimensional_lstm_parameters_one_direction_fully_parallel(
                    self.hidden_states_size, self.input_channels, False, self.use_dropout)
            self.copy_weights_parallel_hidden_and_memory_states_convolution_to_one_directional_mdlstm_parameters(
                mdlstm_parameters_one_direction, i)
            self.copy_parallel_multiple_input_convolutions_computation_to_one_directional_mdlstm_parameters(
                mdlstm_parameters_one_direction, i)
            self.copy_output_gate_memory_state_convolution_to_one_directional_mdlstm_parameters(
                mdlstm_parameters_one_direction, i)
            result.append(mdlstm_parameters_one_direction)
        return result


class MultiDirectionalMultiDimensionalLSTMParametersFullyParallel(
     MultiDimensionalLSTMParametersBase):

    def __init__(self, hidden_states_size, input_channels,
                 use_dropout: bool, number_of_directions: int,
                 parallel_hidden_and_memory_state_column_computation,
                 parallel_input_column_computation,
                 output_gate_memory_state_convolution
                 ):
        super(MultiDirectionalMultiDimensionalLSTMParametersFullyParallel, self).__init__(
            hidden_states_size, input_channels, use_dropout)

        print("Entered MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.init...")

        self.number_of_directions = number_of_directions

        self.parallel_hidden_and_memory_state_column_computation = \
            parallel_hidden_and_memory_state_column_computation

        self.parallel_input_column_computation = \
            parallel_input_column_computation

        # Memory state convolutions
        self.output_gate_memory_state_convolution = output_gate_memory_state_convolution
        nn.init.xavier_uniform_(self.output_gate_memory_state_convolution.weight)

        self.input_columns = None
        self.node_hidden_state_columns = None
        self.node_memory_state_columns = None
        self.previous_memory_state_column = None
        print("multi_dimensional_lstm_parameters."
              "setting self.next_input_column_index to zero...")

        # For some hard to understand reason these values get reset to zero
        # when using (custom) DataParallel (with two gpus) after every example.
        # While this gives the desired behavior, it is not at all clear why this is happening!
        self.next_input_column_index = 0
        # Used to show how this variable "automatically" resets to zero when using
        # data parallel, but not when data parallel is omitted
        # TODO Try to simplify this functionality bug and ask on pytorch forums
        # why this happens
        # self.bladie_input_column_index = 0
        self.skewed_images_variable = None

    def reset_next_input_column_index(self):
        self.next_input_column_index = 0

    @staticmethod
    def create_multi_directional_mdlstm_or_leaky_lp_cell_parameters_fully_parallel(
            hidden_states_size, input_channels, clamp_gradients: bool, use_dropout: bool, number_of_directions: int,
            output_gate_memory_state_convolution):

        parallel_hidden_and_memory_state_column_computation = \
            MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            create_parallel_hidden_and_memory_state_column_computation(
                hidden_states_size, clamp_gradients, use_dropout, number_of_directions)

        parallel_input_column_computation = \
            MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            create_parallel_input_column_computation(hidden_states_size, clamp_gradients,
                                                     use_dropout, number_of_directions,
                                                     input_channels)

        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel(
            hidden_states_size, input_channels, use_dropout, number_of_directions,
            parallel_hidden_and_memory_state_column_computation,
            parallel_input_column_computation, output_gate_memory_state_convolution)

    @staticmethod
    def create_multi_directional_multi_dimensional_lstm_parameters_fully_parallel(
            hidden_states_size, input_channels, clamp_gradients: bool, use_dropout: bool, number_of_directions: int):
        print(">>>create_multi_directional_multi_dimensional_lstm_parameters_fully_parallel...")
        output_gate_memory_state_convolution = nn.Conv1d(
            hidden_states_size * number_of_directions,
            hidden_states_size * number_of_directions, 1,
            groups=number_of_directions)
        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            create_multi_directional_mdlstm_or_leaky_lp_cell_parameters_fully_parallel(
                hidden_states_size, input_channels, clamp_gradients, use_dropout, number_of_directions,
                output_gate_memory_state_convolution)

    @staticmethod
    def create_multi_directional_multi_dimensional_leaky_lp_cell_parameters_fully_parallel(
            hidden_states_size, input_channels, clamp_gradients: bool, use_dropout: bool, number_of_directions: int):
        print(">>>create_multi_directional_multi_dimensional_leaky_lp_cell_parameters_fully_parallel...")
        output_gate_memory_state_convolution = nn.Conv1d(
            hidden_states_size * number_of_directions,
            hidden_states_size * number_of_directions * 2, 1,
            groups=number_of_directions * 2)
        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel. \
            create_multi_directional_mdlstm_or_leaky_lp_cell_parameters_fully_parallel(
                hidden_states_size, input_channels, clamp_gradients, use_dropout, number_of_directions,
                output_gate_memory_state_convolution)

    @staticmethod
    def create_parallel_input_column_computation(
            hidden_states_size: int, clamp_gradients: bool,
            use_dropout: bool, number_of_directions: int, input_channels: int):
        number_of_single_input_weightings_per_group = list([])
        for i in range(0, number_of_directions):
            number_of_single_input_weightings_per_group.extend(list([
                MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                    NUMBER_OF_INPUT_CONVOLUTIONS_PER_DIRECTION
            ]))

        parallel_input_column_computation = \
            ParallelMultipleInputWeightingsComputation. \
            create_parallel_multiple_input_weighting_computation(
                    input_channels, hidden_states_size, number_of_single_input_weightings_per_group,
                    clamp_gradients, use_dropout
                )
        return parallel_input_column_computation

    @staticmethod
    def create_parallel_hidden_and_memory_state_column_computation(
            hidden_states_size: int, clamp_gradients: bool,
            use_dropout: bool, number_of_directions: int):
        number_of_paired_input_weightings_per_group = list([])
        for i in range(0, number_of_directions):
            number_of_paired_input_weightings_per_group.extend(
                list([MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                     NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS,
                      MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                      NUMBER_OF_PAIRED_MEMORY_STATE_WEIGHTINGS]))

        parallel_hidden_and_memory_state_column_computation = \
            ParallelMultipleStateWeightingsComputation.\
            create_parallel_multiple_state_weighting_computation_multiple_groups(
                hidden_states_size, number_of_paired_input_weightings_per_group,
                clamp_gradients, use_dropout)
        return parallel_hidden_and_memory_state_column_computation

    @staticmethod
    def num_paired_hidden_and_memory_state_weightings():
        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS + \
               MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.NUMBER_OF_PAIRED_MEMORY_STATE_WEIGHTINGS

    def get_next_node_hidden_state_columns(self, node_hidden_and_memory_state_columns):

        node_hidden_state_columns_lists = list([])
        for i in range(0, self.number_of_directions):
            offset = MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
                         num_paired_hidden_and_memory_state_weightings() * i
            node_hidden_state_columns = ParallelMultipleStateWeightingsComputation. \
                compute_summed_outputs_every_pair_static(
                    node_hidden_and_memory_state_columns[offset:offset +
                                                                MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                                         NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS])
            node_hidden_state_columns_lists.append(node_hidden_state_columns)

        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            concatenate_elements_list_of_lists_along_dimension(node_hidden_state_columns_lists, 1)

    def get_next_node_memory_state_columns(self, node_hidden_and_memory_state_columns):

        node_memory_state_columns_lists = list([])
        for i in range(0, self.number_of_directions):
            offset = MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
                         num_paired_hidden_and_memory_state_weightings() * i
            node_hidden_state_columns = \
                node_hidden_and_memory_state_columns[offset +
                                                     MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                                     NUMBER_OF_PAIRED_HIDDEN_STATE_WEIGHTINGS
                                                     :offset +
                                                      MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                                     num_paired_hidden_and_memory_state_weightings()
                                                     ]
            node_memory_state_columns_lists.append(node_hidden_state_columns)

        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution. \
            concatenate_elements_list_of_lists_of_tuples_along_dimension(node_memory_state_columns_lists, 1)

    def get_next_input_columns(self, input_columns):

        input_columns_lists = list([])

        number_of_input_convolutions_per_direction = MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution. \
            NUMBER_OF_INPUT_CONVOLUTIONS_PER_DIRECTION

        for i in range(0, self.number_of_directions):
            offset = number_of_input_convolutions_per_direction * i
            input_columns_for_direction = \
                input_columns[offset:offset + number_of_input_convolutions_per_direction]
            input_columns_lists.append(input_columns_for_direction)

        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution. \
            concatenate_elements_list_of_lists_along_dimension(input_columns_lists, 1)

    def prepare_input_convolutions(self, skewed_images_variable):
        self.skewed_images_variable = skewed_images_variable
        # print("self.skewed_images_variable.size(): " +
        #       str(self.skewed_images_variable.size()))
        return

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column,  mask: torch.Tensor):
        # print("Entered MultiDirectionalMultiDimensionalLSTMParametersFullyParallel." +
        #       "prepare_computation_next_column_functions...")

        if previous_hidden_state_column.size(1) != self.hidden_states_size * self.number_of_directions:
            raise RuntimeError("Error: the size of the second dimension of" +
                               " previous_hidden_state_column should match the number of directions" +
                               "times the number of hidden states")

        if previous_memory_state_column.size(1) != self.hidden_states_size * self.number_of_directions:
            raise RuntimeError("Error: the size of the second dimension of" +
                               " memory_state_column should match the number of directions " +
                               "times the number of hidden states")

        previous_hidden_state_columns_split_by_direction = torch.chunk(previous_hidden_state_column,
                                                                       self.number_of_directions, 1)
        previous_memory_state_columns_split_by_direction = torch.chunk(previous_memory_state_column,
                                                                       self.number_of_directions, 1)
        # print("multi_dimensional_lstm_parameters - self.next_input_column_index: " +
        #           str(self.next_input_column_index))
        # print("multi_dimensional_lstm_parameters - self.bladie_input_column_index: "
        #      + str(self.bladie_input_column_index))
        input_column = self.skewed_images_variable[:, :, :, self.next_input_column_index]

        # print("prepare_computation_next_column_functions - input_column: " + str(input_column))
        input_columns_split_by_direction = torch.chunk(input_column,
                                                       self.number_of_directions, 1)

        computation_arguments_list = list([])
        for i in range(0, self.number_of_directions):
            computation_arguments_list.append(previous_hidden_state_columns_split_by_direction[i])
            computation_arguments_list.append(previous_memory_state_columns_split_by_direction[i])

        node_hidden_and_memory_state_columns = \
            self.parallel_hidden_and_memory_state_column_computation.\
            compute_result_and_split_into_pairs_with_second_pair_element_shifted_multiple_groups(
                computation_arguments_list,
                mask)

        input_convolution_result_columns = \
            self.parallel_input_column_computation.\
            compute_result_and_split_into_columns_multiple_groups(
                    input_columns_split_by_direction)

        # Sanity check that the number of output pairs is as expected
        if len(node_hidden_and_memory_state_columns) != \
                (MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                    num_paired_hidden_and_memory_state_weightings() * self.number_of_directions):
            raise RuntimeError("Error: there are " + str(self.number_of_directions) + " directions, " +
                               "therefore expected " +
                               str(MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.
                                   num_paired_hidden_and_memory_state_weightings()) + " * " +
                               str(self.number_of_directions) + " output pairs, but got" +
                               str(len(node_hidden_and_memory_state_columns)))

        self.node_hidden_state_columns = self.get_next_node_hidden_state_columns(node_hidden_and_memory_state_columns)
        self.node_memory_state_columns = self.get_next_node_memory_state_columns(node_hidden_and_memory_state_columns)
        self.input_columns = self.get_next_input_columns(input_convolution_result_columns)

        self.previous_memory_state_column = previous_memory_state_column

        # print("finished prepare_computation_next_column_functions")

        # Increment the next input column index
        self.next_input_column_index += 1
        # self.bladie_input_column_index += 1

    def compute_output_gate_memory_state_weighted_input(self, previous_memory_state_column):
        if TensorUtils.number_of_dimensions(previous_memory_state_column) != 3:
            raise RuntimeError("Error: prepare_input_convolution requires 3 dimensional input"
                               + " got size: " + str(previous_memory_state_column.size()))

        if previous_memory_state_column.size(1) != self.hidden_states_size * self.number_of_directions:
            raise RuntimeError("Error: the size of the first dimension of" +
                               " memory_state_column should match the number of directions" +
                               " times the number of hidden states")

        # previous_memory_state_columns_split_by_direction = torch.split(previous_memory_state_column, 1, 0)
        # previous_memory_state_column_catted_on_channel_dimension = \
        #     torch.cat(previous_memory_state_columns_split_by_direction, 1)
        #
        # result_catted_on_channel_dimension = StateUpdateBlock.compute_weighted_state_input_state_one(
        #     self.output_gate_memory_state_convolution,
        #     previous_memory_state_column_catted_on_channel_dimension)
        # # print("result_catted_on_channel_dimension.size(): " + str(result_catted_on_channel_dimension.size()))
        # result_split_into_directions = torch.chunk(result_catted_on_channel_dimension, self.number_of_directions, 1)
        # # Re-concatenate the direction results on the batch dimension
        # result = torch.cat(result_split_into_directions, 0)
        # # print("result.size(): " + str(result.size()))
        # return result

        result_catted_on_channel_dimension = StateUpdateBlock.compute_weighted_state_input_state_one(
            self.output_gate_memory_state_convolution,
            previous_memory_state_column)

        return result_catted_on_channel_dimension

    def get_input_input_column(self, column_index):
        result = self.input_columns[0]
        # print("get_input_input_colum - result" + str(result))
        # print("get_input_input_colum - result.size()" + str(result.size()))
        return result

    def get_input_gate_input_column(self, column_index):
        return self.input_columns[1]

    def get_forget_gate_one_input_column(self, column_index):
        return self.input_columns[2]

    def get_forget_gate_two_input_column(self, column_index):
        return self.input_columns[3]

    def get_output_gate_input_column(self, column_index):
        return self.input_columns[4]

    def get_input_hidden_state_column(self):
        return self.node_hidden_state_columns[0]

    def get_input_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[1]

    def get_forget_gate_one_hidden_state_column(self):
        return self.node_hidden_state_columns[2]

    def get_forget_gate_two_hidden_state_column(self):
        return self.node_hidden_state_columns[3]

    def get_output_gate_hidden_state_column(self):
        return self.node_hidden_state_columns[4]

    def get_input_gate_memory_state_column(self):
        input_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[0]
        input_gate_memory_state_column = input_gate_memory_state_column_part_pair[0] + \
            input_gate_memory_state_column_part_pair[1]
        return input_gate_memory_state_column

    def get_forget_gate_one_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[0]
        return forget_gate_memory_state_column

    def get_forget_gate_two_memory_state_column(self):
        forget_gate_memory_state_column_part_pair = \
            self.node_memory_state_columns[1]
        forget_gate_memory_state_column = forget_gate_memory_state_column_part_pair[1]
        return forget_gate_memory_state_column

    def number_of_hidden_and_memory_state_weights_per_direction(self):
        return self.hidden_states_size * 2 * MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            num_paired_hidden_and_memory_state_weightings()

    def number_of_input_weighting_weights_per_direction(self):
        return self.hidden_states_size * \
               MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
               NUMBER_OF_INPUT_CONVOLUTIONS_PER_DIRECTION

    def set_bias_forget_gates_hidden_states_input(self):
        relative_start_index = self.hidden_states_size * 2 * 2
        relative_end_index = self.hidden_states_size * 2 * 4
        # print("start index: " + str(relative_start_index) + " end index: " + str(relative_end_index))

        for direction_index in range(0, self.number_of_directions):
            offset = self.number_of_hidden_and_memory_state_weights_per_direction() * direction_index

            for index in range(offset + relative_start_index, offset + relative_end_index):
                self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.bias.data[index] = \
                    OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
            # print("after: self.parallel_hidden_state_column_computation.parallel_convolution.bias.data: " +
            #      str(self.parallel_hidden_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_memory_states_input(self):
        # self.forget_gate_one_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_memory_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # print("before: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #        str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))
        relative_start_index = self.hidden_states_size * 2 * 6
        relative_end_index = self.number_of_hidden_and_memory_state_weights_per_direction()
        # print("start index: " + str(relative_start_index) + " end index: " + str(relative_end_index))

        for direction_index in range(0, self.number_of_directions):
            offset = self.number_of_hidden_and_memory_state_weights_per_direction() * direction_index

            for index in range(offset + relative_start_index, offset + relative_end_index):
                self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.bias.data[index] = \
                    OneDirectionalMultiDimensionalLSTMParametersBase.FORGET_GATE_BIAS_INIT
            # print("after: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
            #      str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))

    def set_bias_everything_to_zero(self):
        """
        For Leaky LP Cell networks, rather than doing complicated initialization of some of
        the gate biases to one, simply set all bias values to zero.
        Importantly, for Leaky LP cells, initializing bias for lambda gates to one is a
        bad idea. Since a lambda gate is a switch, and a-priori both switch options should
        be equally likely, so bias zero is appropriate for such gates. A bias of one,
        tells that one of the outputs of the switch is preferred strongly, but there is
        no ground for that. Initialiation to one only makes sense for normal (MD)LSTM
        forget gates.
        )
        :return: Nothing, the bias is set in place
        """

        print(">>> Multi_dimensional_lstm_parameters: Initializing the bias of everything to zero!")
        # Set bias to zero
        with torch.no_grad():
            self.output_gate_memory_state_convolution.bias.zero_()
            self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.bias.zero_()

    def set_bias_forget_gates_to_one(self):
        print(">>> Multi_dimensional_lstm_parameters: Set bias forget gates to one")
        # self.set_bias_forget_gates_image_input()
        self.set_bias_forget_gates_memory_states_input()
        self.set_bias_forget_gates_hidden_states_input()

    def set_training(self, training):
        self.parallel_hidden_and_memory_state_column_computation.set_training(training)

    def forward(self, x):
        raise NotImplementedError

    def copy_weights_parallel_hidden_and_memory_states_convolution_to_one_directional_mdlstm_parameters(
            self, mdlstm_parameters_one_direction, direction_index):

        relative_start_index = 0
        relative_end_index = self.number_of_hidden_and_memory_state_weights_per_direction()
        for one_directional_mdlstm_index in range(relative_start_index, relative_end_index):
            multi_directional_mdlstm_index = \
                one_directional_mdlstm_index + \
                self.number_of_hidden_and_memory_state_weights_per_direction() * direction_index
            mdlstm_parameters_one_direction.parallel_hidden_and_memory_state_column_computation.\
                parallel_convolution.bias.data[one_directional_mdlstm_index] =\
                self.parallel_hidden_and_memory_state_column_computation.\
                parallel_convolution.bias.data[multi_directional_mdlstm_index]
            mdlstm_parameters_one_direction.parallel_hidden_and_memory_state_column_computation.\
                parallel_convolution.weight.data[one_directional_mdlstm_index, :, :] = \
                self.parallel_hidden_and_memory_state_column_computation.parallel_convolution.\
                weight.data[multi_directional_mdlstm_index, :, :]

    def copy_parallel_multiple_input_convolutions_computation_to_one_directional_mdlstm_parameters(
            self, mdlstm_parameters_one_direction, direction_index):

        # print(">>> copy_parallel_multiple_input_convolutions_computation_to_one_directional_mdlstm_parameters...")

        # mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation = \
        #                 self.parallel_multiple_input_convolutions_computations[direction_index]

        # print(">>> direction_index: " + str(direction_index))
        # print(">>>  self.number_of_input_weighting_weights_per_direction(): "
        #       + str(self.number_of_input_weighting_weights_per_direction()))

        relative_start_index = 0
        relative_end_index = self.number_of_input_weighting_weights_per_direction()
        for one_directional_mdlstm_index in range(relative_start_index, relative_end_index):
            multi_directional_mdlstm_index = \
                one_directional_mdlstm_index + \
                self.number_of_input_weighting_weights_per_direction() * direction_index
            mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
                parallel_convolution.bias.data[one_directional_mdlstm_index] = \
                self.parallel_input_column_computation.\
                parallel_convolution.bias.data[multi_directional_mdlstm_index]

            # print("one_directional_mdlstm_index: " + str(one_directional_mdlstm_index))
            # print("multi_directional_mdlstm_index: " + str(multi_directional_mdlstm_index))
            #
            # print(" mdlstm_parameters_one_direction.parallel_hidden_and_memory_state_column_computation.\
            #                         parallel_convolution.weight.data.size()" +
            #       str(mdlstm_parameters_one_direction.parallel_hidden_and_memory_state_column_computation.\
            #           parallel_convolution.weight.data.size()))
            # print(" mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
            #                 parallel_convolution.weight.data.size()" +
            #       str(mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation. \
            #           parallel_convolution.weight.data.size()))
            # print(" mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
            #     parallel_convolution.weight.data[one_directional_mdlstm_index, :, :].size()" +
            #       str( mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
            #     parallel_convolution.weight.data[one_directional_mdlstm_index, :, :].size()))
            # print("self.parallel_input_and_hidden_and_memory_state_column_computation.parallel_convolution. \
            #     weight.data[multi_directional_mdlstm_index, :, :].size(): " +
            #       str(self.parallel_hidden_and_memory_state_column_computation.parallel_convolution. \
            #         weight.data[multi_directional_mdlstm_index, :, :].size()))

            # print(" mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
            #     parallel_convolution.weight.data[begin_index:end_index, :, :, :].size()" +
            #       str(mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
            #     parallel_convolution.weight.data[begin_index:end_index, :, :, :].size()))

            # print("self.parallel_input_column_computation.parallel_convolution.\
            #     weight.data.size(): " + str(self.parallel_input_column_computation.
            #                                 parallel_convolution.weight.data.size()))

            mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation.\
                parallel_convolution.weight.data[one_directional_mdlstm_index, :, 0, 0] = \
                self.parallel_input_column_computation.parallel_convolution. \
                weight.data[multi_directional_mdlstm_index, :, :]

    def copy_output_gate_memory_state_convolution_to_one_directional_mdlstm_parameters(
            self, mdlstm_parameters_one_direction, direction_index):
        relative_start_index = 0
        relative_end_index = self.hidden_states_size
        for one_directional_mdlstm_index in range(relative_start_index, relative_end_index):
            multi_directional_mdlstm_index = \
                self.hidden_states_size * direction_index + one_directional_mdlstm_index

            mdlstm_parameters_one_direction.output_gate_memory_state_convolution.bias[one_directional_mdlstm_index] = \
                self.output_gate_memory_state_convolution.bias[multi_directional_mdlstm_index]

            mdlstm_parameters_one_direction.output_gate_memory_state_convolution.\
                weight[one_directional_mdlstm_index, :, :] = \
                self.output_gate_memory_state_convolution.weight[multi_directional_mdlstm_index, :, :]

    """
    This methods extracts/creates  a list of one-directional MDLSTM parameters based on the
    current weights of the multi-directional MDLSTM parameters. This is mainly useful for 
    testing purposes. Creating (one-directional) MDLSTMs from the thus extracted parameters, 
    it can be tested whether separately executed one-directional MDLSTMs produce the same results
    as the corresponding outputs for that direction of the multi-directional MDLSTM from which 
    the parameters were taken. This kind of testing is useful, since it is very hard to test 
    directly whether the multi-directional MDLSTM "works". This is actually a general problem 
    for testing neural networks: finding and fixing problems with mismatching tensor sizes etc
    is easy enough. But once something runs without errors, it can be very hard to determine 
    whether it is actually computing the right thing. 
    """
    def create_one_directional_mdlstm_parameters_each_direction_using_current_weights(self):
        result = list([])

        for i in range(0, self.number_of_directions):
            mdlstm_parameters_one_direction = MultiDimensionalLSTMParametersOneDirectionFullyParallel.\
                create_multi_dimensional_lstm_parameters_one_direction_fully_parallel(
                    self.hidden_states_size, self.input_channels, False, self.use_dropout)
            self.copy_weights_parallel_hidden_and_memory_states_convolution_to_one_directional_mdlstm_parameters(
                mdlstm_parameters_one_direction, i)
            self.copy_parallel_multiple_input_convolutions_computation_to_one_directional_mdlstm_parameters(
                mdlstm_parameters_one_direction, i)
            self.copy_output_gate_memory_state_convolution_to_one_directional_mdlstm_parameters(
                mdlstm_parameters_one_direction, i)
            result.append(mdlstm_parameters_one_direction)
        return result


class MultiDimensionalLSTMParametersCreator:

    # Needs to be implemented in the subclasses
    @abstractmethod
    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels,
                                                               use_dropout: bool,
                                                               clamp_gradients: bool):
        raise RuntimeError("not implemented")

    @abstractmethod
    def create_multi_directional_multi_dimensional_lstm_parameters(self,
                                                                   hidden_states_size, input_channels,
                                                                   use_dropout: bool,
                                                                   clamp_gradients: bool,
                                                                   number_of_directions: bool):
        raise RuntimeError("not implemented")


class MultiDirectionalMultiDimensionalLSTMParametersCreatorFullyParallel(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels,
                                                               clamp_gradients: bool,
                                                               use_dropout: bool,
                                                               ):
        raise RuntimeError("not implemented")

    @abstractmethod
    def create_multi_directional_multi_dimensional_lstm_parameters(self,
                                                                   hidden_states_size,
                                                                   input_channels,
                                                                   use_dropout: bool,
                                                                   clamp_gradients: bool,
                                                                   number_of_directions: bool):
        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            create_multi_directional_multi_dimensional_lstm_parameters_fully_parallel(
                hidden_states_size, input_channels, clamp_gradients, use_dropout, number_of_directions)


class MultiDirectionalMultiDimensionalLeakyLPCellParametersCreatorFullyParallel(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels,
                                                               clamp_gradients: bool,
                                                               use_dropout: bool,
                                                               ):
        raise RuntimeError("not implemented")

    @abstractmethod
    def create_multi_directional_multi_dimensional_lstm_parameters(self,
                                                                   hidden_states_size,
                                                                   input_channels,
                                                                   use_dropout: bool,
                                                                   clamp_gradients: bool,
                                                                   number_of_directions: bool):
        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            create_multi_directional_multi_dimensional_leaky_lp_cell_parameters_fully_parallel(
                hidden_states_size, input_channels, clamp_gradients, use_dropout, number_of_directions)




class MultiDirectionalMultiDimensionalLSTMParametersCreatorParallelWithSeparateInputConvolution(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels,
                                                               clamp_gradients: bool,
                                                               use_dropout: bool,
                                                               ):
        raise RuntimeError("not implemented")

    @abstractmethod
    def create_multi_directional_multi_dimensional_lstm_parameters(self,
                                                                   hidden_states_size,
                                                                   input_channels,
                                                                   use_dropout: bool,
                                                                   clamp_gradients: bool,
                                                                   number_of_directions: bool):
        return MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution.\
            create_multi_directional_multi_dimensional_lstm_parameters_parallel_with_separate_input_convolution(
                hidden_states_size, input_channels, clamp_gradients, use_dropout, number_of_directions)


class MultiDimensionalLSTMParametersCreatorFullyParallel(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels,
                                                               clamp_gradients: bool,
                                                               use_dropout: bool,
                                                               ):
        return MultiDimensionalLSTMParametersOneDirectionFullyParallel.\
            create_multi_dimensional_lstm_parameters_one_direction_fully_parallel(
                hidden_states_size, input_channels, clamp_gradients, use_dropout)

    @abstractmethod
    def create_multi_directional_multi_dimensional_lstm_parameters(self,
                                                                   hidden_states_size, input_channels,
                                                                   use_dropout: bool,
                                                                   clamp_gradients: bool,
                                                                   number_of_directions: bool):
        raise RuntimeError("not implemented")


class MultiDimensionalLSTMParametersCreatorFast(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels,
                                                               clamp_gradients: bool,
                                                               use_dropout: bool,
                                                               ):
        return MultiDimensionalLSTMParametersOneDirectionFast.\
            create_multi_dimensional_lstm_parameters_one_direction_fast(hidden_states_size, input_channels,
                                                                        clamp_gradients, use_dropout)

    @abstractmethod
    def create_multi_directional_multi_dimensional_lstm_parameters(self,
                                                                   hidden_states_size, input_channels,
                                                                   use_dropout: bool,
                                                                   clamp_gradients: bool,
                                                                   number_of_directions: bool):
        raise RuntimeError("not implemented")


class MultiDimensionalLSTMParametersCreatorSlow(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self, hidden_states_size, input_channels,
                                                               clamp_gradients: bool,
                                                               use_dropout: bool):
        # TODO: clamp_gradient is not implemented for the slow version
        return MultiDimensionalLSTMParametersOneDirection.\
            create_multi_dimensional_lstm_parameters_one_direction(hidden_states_size, input_channels,
                                                                   use_dropout)

    @ abstractmethod
    def create_multi_directional_multi_dimensional_lstm_parameters(self,
                                                                   hidden_states_size, input_channels,
                                                                   use_dropout: bool,
                                                                   clamp_gradients: bool,
                                                                   number_of_directions: bool):
        raise RuntimeError("not implemented")

