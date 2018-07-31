from modules.state_update_block import StateUpdateBlock
from modules.parallel_multiple_state_weightings_computation import ParallelMultipleStateWeightingsComputation
from modules.parallel_multiple_input_convolutions_computation import ParallelMultipleInputConvolutionsComputation
from abc import abstractmethod
import torch.nn as nn
from torch.nn.modules.module import Module
import torch
from util.tensor_utils import TensorUtils


class MultiDimensionalLSTMParametersBase(Module):
    # https://github.com/pytorch/pytorch/issues/750
    FORGET_GATE_BIAS_INIT = 1  # Good for normal LSTM
    # FORGET_GATE_BIAS_INIT = 0    # For stable learning in MDLSTM ? Doesn't really seem to help to avoid nans

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

    @abstractmethod
    def cleanup_input_convolution_results(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        raise RuntimeError("not implemented")

    @abstractmethod
    def get_input_input_matrix(self):
        raise RuntimeError("not implemented")

    @abstractmethod
    def get_input_gate_input_matrix(self):
        raise RuntimeError("not implemented")

    @abstractmethod
    def get_forget_gate_one_input_matrix(self):
        raise RuntimeError("not implemented")

    @abstractmethod
    def get_forget_gate_two_input_matrix(self):
        raise RuntimeError("not implemented")

    @abstractmethod
    def get_output_gate_input_matrix(self):
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
    def get_all_parameters_as_list(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def set_bias_forget_gates_to_one(self):
        raise RuntimeError("not implemented")

    # Needs to be implemented in the subclasses
    @abstractmethod
    def set_training(self, training):
        raise RuntimeError("not implemented")

    @abstractmethod
    def get_all_input_convolutions_as_list(self):
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


class OneDirectionalMultiDimensionalLSTMParametersBase(MultiDimensionalLSTMParametersBase):
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

    def cleanup_input_convolution_results(self):
        return

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        self.previous_hidden_state_column = previous_hidden_state_column
        self.previous_memory_state_column = previous_memory_state_column

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

    def get_all_hidden_state_convolutions_as_list(self):
        result = list([])
        result.extend(self.input_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.input_gate_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.forget_gate_one_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.forget_gate_two_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.output_gate_hidden_state_update_block.get_state_convolutions_as_list())
        return result

    def get_all_memory_state_convolutions_as_list(self):
        result = list([])
        result.append(self.output_gate_memory_state_convolution)
        result.append(self.forget_gate_one_memory_state_convolution)
        result.append(self.forget_gate_two_memory_state_convolution)
        result.extend(self.input_gate_memory_state_update_block.get_state_convolutions_as_list())
        return result

    def get_all_parameters_as_list(self):
        result = list([])
        result.extend(self.get_all_input_convolutions_as_list())
        result.extend(self.get_all_memory_state_convolutions_as_list())
        result.extend(self.get_all_hidden_state_convolutions_as_list())

        print(">>> number of convolution parameter blocks: " + str(len(result)))
        return result

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

    def set_training(self, training):

        # TODO: implement this
        return

    def get_all_input_convolutions_as_list(self):
        result = list([])
        result.append(self.input_input_convolution)
        result.append(self.input_gate_input_convolution)
        result.append(self.forget_gate_one_input_convolution)
        result.append(self.forget_gate_two_input_convolution)
        result.append(self.output_gate_input_convolution)
        return result

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

    def prepare_input_convolutions(self, skewed_images_variable):
        self.input_matrices = self.parallel_multiple_input_convolutions_computation.\
            compute_result_and_split_into_output_elements(skewed_images_variable)

    def cleanup_input_convolution_results(self):
        # Reset the value to None, so that the memory can be cleared,
        # if there are no other users of these results
        self.input_matrices = None

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

    def get_all_parameters_as_list(self):
        result = list([])
        result.extend(self.get_all_input_convolutions_as_list())
        result.append(self.output_gate_memory_state_convolution)
        result.extend(self.parallel_hidden_state_column_computation.get_state_convolutions_as_list())
        result.extend(self.parallel_memory_state_column_computation.get_state_convolutions_as_list())
        # print(">>> number of convolution parameter blocks: " + str(len(result)))
        return result

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

    def set_training(self, training):
        self.parallel_hidden_state_column_computation.set_training(training)
        self.parallel_memory_state_column_computation.set_training(training)

    def get_all_input_convolutions_as_list(self):
        result = list([])
        result.append(self.parallel_multiple_input_convolutions_computation.parallel_convolution)
        return result

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

    def cleanup_input_convolution_results(self):
        # Reset the value to None, so that the memory can be cleared,
        # if there are no other users of these results
        self.input_matrices = None

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

    def get_all_parameters_as_list(self):
        result = list([])
        result.extend(self.get_all_input_convolutions_as_list())
        result.append(self.output_gate_memory_state_convolution)
        result.extend(self.parallel_hidden_and_memory_state_column_computation.get_state_convolutions_as_list())
        # print(">>> number of convolution parameter blocks: " + str(len(result)))
        return result

    def set_bias_forget_gates_memory_states_input(self):
        # self.forget_gate_one_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_one_memory_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # print("before: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #        str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))
        start_index = self.hidden_states_size * 2 * 6
        end_index = self.hidden_states_size * 2 * 7
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

    def set_training(self, training):
        self.parallel_hidden_and_memory_state_column_computation.set_training(training)

    def get_all_input_convolutions_as_list(self):
        result = list([])
        result.append(self.parallel_multiple_input_convolutions_computation.parallel_convolution)
        return result

    def forward(self, x):
        raise NotImplementedError


class MultiDirectionalMultiDimensionalLSTMParametersFullyParallel(MultiDimensionalLSTMParametersBase):
    def __init__(self, hidden_states_size, input_channels,
                 use_dropout: bool, number_of_directions: int,
                 parallel_multiple_input_convolutions_computations,
                 parallel_hidden_and_memory_state_column_computation

                 ):
        super(MultiDirectionalMultiDimensionalLSTMParametersFullyParallel, self).__init__(hidden_states_size,
                                                                                          input_channels,
                                                                                          use_dropout)
        self.number_of_directions = number_of_directions

        self.parallel_multiple_input_convolutions_computations = parallel_multiple_input_convolutions_computations

        self.parallel_hidden_and_memory_state_column_computation = parallel_hidden_and_memory_state_column_computation

        # Memory state convolutions
        self.output_gate_memory_state_convolution = nn.Conv1d(self.hidden_states_size * number_of_directions,
                                                              self.hidden_states_size * number_of_directions, 1,
                                                              groups=number_of_directions)
        nn.init.xavier_uniform_(self.output_gate_memory_state_convolution.weight)

        self.input_matrices = None
        self.node_hidden_state_columns = None
        self.node_memory_state_columns = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_directional_multi_dimensional_lstm_parameters_fully_parallel(
            hidden_states_size, input_channels, clamp_gradients: bool, use_dropout: bool, number_of_directions: int):

        parallel_multiple_input_convolutions_computations = \
            MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            create_parallel_multiple_input_convolution_computations(input_channels, hidden_states_size,
                                                                    clamp_gradients, use_dropout, number_of_directions)

        parallel_hidden_and_memory_state_column_computation = \
            MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            create_parallel_hidden_and_memory_state_column_computation(hidden_states_size, clamp_gradients,
                                                                       use_dropout, number_of_directions)

        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel(
            hidden_states_size, input_channels, use_dropout, number_of_directions,
            parallel_multiple_input_convolutions_computations,
            parallel_hidden_and_memory_state_column_computation)

    @staticmethod
    def create_parallel_hidden_and_memory_state_column_computation(
            hidden_states_size: int, clamp_gradients: bool,
            use_dropout: bool, number_of_directions: int):
        number_of_paired_input_weightings_per_group = list([])
        for i in range(0, number_of_directions):
            number_of_paired_input_weightings_per_group.extend(list([5, 2]))

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
                create_parallel_multiple_input_convolutions_computation(input_channels,
                                                                        hidden_states_size,
                                                                        5,
                                                                        clamp_gradients,
                                                                        use_dropout)
            parallel_multiple_input_convolutions_computations.append(parallel_multiple_input_convolution)
        return parallel_multiple_input_convolutions_computations

    def prepare_input_convolutions(self, skewed_images_variable):
        # print("Entered MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.prepare_input_convolutions...")

        if TensorUtils.number_of_dimensions(skewed_images_variable) != 4:
            raise RuntimeError("Error: prepare_input_convolution requires 4 dimensional input")

        if skewed_images_variable.size(0) != self.number_of_directions:
            raise RuntimeError("Error: the size of the first dimension should match the number of directions")

        # print("MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.prepare_input_convolutions - split")
        # print("skewed_images_variable.size(): " + str(skewed_images_variable.size()))
        # First split the image tensor to get the images for the different directions
        skewed_images_variable_list = torch.split(skewed_images_variable, 1, 0)

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

        input_matrices_lists_grouped_by_index_concatenated = \
            MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            concatenate_elements_list_of_lists_along_dimension(input_matrices_lists, 0)

        # print(">>> len(input_matrices_lists_grouped_by_index_concatenated): " +
        #       str(len(input_matrices_lists_grouped_by_index_concatenated)))
        self.input_matrices = input_matrices_lists_grouped_by_index_concatenated

        if len(self.input_matrices) != 5:
            raise RuntimeError("Error: length of input_matrices after merging matrices for multiple" +
                               " directions should still be 5")

        # Finally concatenate back the input matrices

    def cleanup_input_convolution_results(self):
        # Reset the value to None, so that the memory can be cleared,
        # if there are no other users of these results
        self.input_matrices = None

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

    def get_next_node_hidden_state_columns(self, node_hidden_and_memory_state_columns):

        node_hidden_state_columns_lists = list([])
        for i in range(0, self.number_of_directions):
            offset = 7 * i
            node_hidden_state_columns = ParallelMultipleStateWeightingsComputation. \
                compute_summed_outputs_every_pair_static(node_hidden_and_memory_state_columns[offset:offset+5])
            node_hidden_state_columns_lists.append(node_hidden_state_columns)

        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
            concatenate_elements_list_of_lists_along_dimension(node_hidden_state_columns_lists, 0)

    def get_next_node_memory_state_columns(self, node_hidden_and_memory_state_columns):

        node_memory_state_columns_lists = list([])
        for i in range(0, self.number_of_directions):
            offset = 7 * i
            node_hidden_state_columns = node_hidden_and_memory_state_columns[offset+5:offset+7]
            node_memory_state_columns_lists.append(node_hidden_state_columns)

        return MultiDirectionalMultiDimensionalLSTMParametersFullyParallel. \
            concatenate_elements_list_of_lists_of_tuples_along_dimension(node_memory_state_columns_lists, 0)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column,  mask: torch.Tensor):
        # print("Entered MultiDirectionalMultiDimensionalLSTMParametersFullyParallel." +
        #       "prepare_computation_next_column_functions...")

        if previous_hidden_state_column.size(0) != self.number_of_directions:
            raise RuntimeError("Error: the size of the first dimension of" +
                               " previous_hidden_state_column should match the number of directions")

        if previous_memory_state_column.size(0) != self.number_of_directions:
            raise RuntimeError("Error: the size of the first dimension of" +
                               " memory_state_column should match the number of directions")

        previous_hidden_state_columns_split_by_direction = torch.split(previous_hidden_state_column, 1, 0)
        previous_memory_state_columns_split_by_direction = torch.split(previous_memory_state_column, 1, 0)

        computation_arguments_list = list([])
        for i in range(0, self.number_of_directions):
            computation_arguments_list.append(previous_hidden_state_columns_split_by_direction[i])
            computation_arguments_list.append(previous_memory_state_columns_split_by_direction[i])

        node_hidden_and_memory_state_columns = \
            self.parallel_hidden_and_memory_state_column_computation.\
            compute_result_and_split_into_pairs_with_second_pair_element_shifted_multiple_groups(
                computation_arguments_list, mask)

        # Sanity check that the number of output pairs is as expected
        if len(node_hidden_and_memory_state_columns) != (7 * self.number_of_directions):
            raise RuntimeError("Error: there are " + str(self.number_of_directions) + " directions, " +
                               "therefore expected " + 7 + " * " + str(self.number_of_directions) +
                               " output pairs")

        self.node_hidden_state_columns = self.get_next_node_hidden_state_columns(node_hidden_and_memory_state_columns)
        self.node_memory_state_columns = self.get_next_node_memory_state_columns(node_hidden_and_memory_state_columns)

        self.previous_memory_state_column = previous_memory_state_column

        # print("finished prepare_computation_next_column_functions")

    def compute_output_gate_memory_state_weighted_input(self, previous_memory_state_column):
        if TensorUtils.number_of_dimensions(previous_memory_state_column) != 3:
            raise RuntimeError("Error: prepare_input_convolution requires 3 dimensional input"
                               + " got size: " + str(previous_memory_state_column.size()))

        if previous_memory_state_column.size(0) != self.number_of_directions:
            raise RuntimeError("Error: the size of the first dimension of" +
                               " memory_state_column should match the number of directions")

        previous_memory_state_columns_split_by_direction = torch.split(previous_memory_state_column, 1, 0)
        previous_memory_state_column_catted_on_channel_dimension = \
            torch.cat(previous_memory_state_columns_split_by_direction, 1)

        result_catted_on_channel_dimension = StateUpdateBlock.compute_weighted_state_input_state_one(
            self.output_gate_memory_state_convolution,
            previous_memory_state_column_catted_on_channel_dimension)
        # print("result_catted_on_channel_dimension.size(): " + str(result_catted_on_channel_dimension.size()))
        result_split_into_directions = torch.chunk(result_catted_on_channel_dimension, self.number_of_directions, 1)
        # Re-concatenate the direction results on the batch dimension
        result = torch.cat(result_split_into_directions, 0)
        # print("result.size(): " + str(result.size()))
        return result

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

    def get_all_parameters_as_list(self):
        result = list([])
        result.extend(self.get_all_input_convolutions_as_list())
        result.append(self.output_gate_memory_state_convolution)
        result.extend(self.parallel_hidden_and_memory_state_column_computation.get_state_convolutions_as_list())
        # print(">>> number of convolution parameter blocks: " + str(len(result)))
        return result

    def number_of_hidden_and_memory_state_weights_per_direction(self):
        return self.hidden_states_size * 2 * 7

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

    def set_training(self, training):
        self.parallel_hidden_and_memory_state_column_computation.set_training(training)

    def get_all_input_convolutions_as_list(self):
        result = list([])
        for input_convolution_computation in self.parallel_multiple_input_convolutions_computations:
            result.append(input_convolution_computation.parallel_convolution)
        return result

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
        mdlstm_parameters_one_direction.parallel_multiple_input_convolutions_computation =\
            self.parallel_multiple_input_convolutions_computations[direction_index]

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

