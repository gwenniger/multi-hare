from modules.state_update_block import StateUpdateBlock
from modules.parallel_multiple_state_weightings_computation import ParallelMultipleStateWeightingsComputation
from abc import abstractmethod
import torch.nn as nn
from torch.nn.modules.module import Module


class MultiDimensionalLSTMParametersOneDirectionBase(Module):
    # https://github.com/pytorch/pytorch/issues/750
    FORGET_GATE_BIAS_INIT = 1

    def __init__(self, hidden_states_size, input_channels, use_dropout):
        super(MultiDimensionalLSTMParametersOneDirectionBase, self).__init__()
        self.input_channels = input_channels
        self.hidden_states_size = hidden_states_size
        self.use_dropout = use_dropout

        # Input convolutions
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

        # Memory state convolutions
        self.output_gate_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                              self.hidden_states_size, 1)

        # TODO: Add dropout to the remaining layers

    def set_bias_forget_gates_image_input(self):
        self.forget_gate_one_input_convolution.bias.data.fill_(
            MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT
        )
        self.forget_gate_two_input_convolution.bias.data.fill_(
            MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT
        )

    # Needs to be implemented in the subclasses
    @abstractmethod
    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        raise RuntimeError("not implemented")

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

    def get_all_input_convolutions_as_list(self):
        result = list([])
        result.append(self.input_input_convolution)
        result.append(self.input_gate_input_convolution)
        result.append(self.forget_gate_one_input_convolution)
        result.append(self.forget_gate_two_input_convolution)
        result.append(self.output_gate_input_convolution)
        return result

    # This class extends Module so as to make sure that the parameters
    # are properly copied (to the right cuda device) when using nn.DataParallel(model)
    # and the to(device) method from  the Module base class
    # http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html
    # The class is not however meant to be used as a stand-alone Module, so forward
    # is not implemented
    def forward(self, x):
        raise NotImplementedError


class MultiDimensionalLSTMParametersOneDirection(MultiDimensionalLSTMParametersOneDirectionBase):
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

        self.previous_hidden_state_column = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction(hidden_states_size, input_channels, use_dropout: bool):
        return MultiDimensionalLSTMParametersOneDirection(hidden_states_size, input_channels, use_dropout)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        self.previous_hidden_state_column = previous_hidden_state_column
        self.previous_memory_state_column = previous_memory_state_column

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
            MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT)

        self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(
            MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT
        )

        # self.forget_gate_two_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        # self.forget_gate_two_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        self.forget_gate_two_memory_state_convolution.bias.data.fill_(
            MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT)

        self.forget_gate_two_hidden_state_update_block.set_bias_for_convolutions(
            MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT
        )

        # self.set_bias_forget_gates_image_input()

    def set_training(self, training):

        # TODO: implement this
        return

    def forward(self, x):
        raise NotImplementedError


    # This implementation of MultiDimensionalLSTMParametersOneDirectionBase uses special 1d convolutions wrapped by the
# ParallelMultipleStateWeightingsComputation to perform several (N) 1d convolutions over the same input together, using
# a single convolution with N times as many outputs
# https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360
class MultiDimensionalLSTMParametersOneDirectionFast(MultiDimensionalLSTMParametersOneDirectionBase):
    def __init__(self, hidden_states_size, input_channels, use_dropout: bool):
        super(MultiDimensionalLSTMParametersOneDirectionFast, self).__init__(hidden_states_size, input_channels,
                                                                             use_dropout)

        # There are five paired input weightings, in this case, pairs of previous hidden state
        # inputs, namely for: 1) the input, 2) the input gate, 3) forget gate one,
        # 4) forget gate two, 5) the output gate
        self.parallel_hidden_state_column_computation = ParallelMultipleStateWeightingsComputation.create_parallel_multiple_state_weighting_computation(
            hidden_states_size, 5, use_dropout)

        self.parallel_memory_state_column_computation = ParallelMultipleStateWeightingsComputation.create_parallel_multiple_state_weighting_computation(
            hidden_states_size, 2, use_dropout)

        self.node_hidden_state_columns = None
        self.node_memory_state_columns = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction_fast(hidden_states_size, input_channels,
                                                                    use_dropout: bool):
        return MultiDimensionalLSTMParametersOneDirectionFast(hidden_states_size, input_channels, use_dropout)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        # The hidden state columns for the different computational nodes:
        # 1) the input, 2) the input gate, 3) forget gate one,
        # 4) forget gate two, 5) the output gate
        self.node_hidden_state_columns = \
            self.parallel_hidden_state_column_computation.compute_summed_outputs_every_pair(previous_hidden_state_column)
        self.previous_memory_state_column = previous_memory_state_column

        self.node_memory_state_columns = self.\
            parallel_memory_state_column_computation.\
            compute_result_and_split_into_pairs_with_second_pair_element_shifted(previous_memory_state_column)

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
                MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT
        # print("after: self.parallel_memory_state_column_computation.parallel_convolution.bias.data: " +
        #      str(self.parallel_memory_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_hidden_states_input(self):
        start_index = self.hidden_states_size * 2 * 2
        end_index = self.hidden_states_size * 2 * 4
        # print("start index: " + str(start_index) + " end index: " + str(end_index))
        for index in range(start_index, end_index):
            self.parallel_hidden_state_column_computation.parallel_convolution.bias.data[index] = \
                MultiDimensionalLSTMParametersOneDirectionBase.FORGET_GATE_BIAS_INIT
        # print("after: self.parallel_hidden_state_column_computation.parallel_convolution.bias.data: " +
        #      str(self.parallel_hidden_state_column_computation.parallel_convolution.bias.data))

    def set_bias_forget_gates_to_one(self):
        # self.set_bias_forget_gates_image_input()
        self.set_bias_forget_gates_memory_states_input()
        self.set_bias_forget_gates_hidden_states_input()

    def set_training(self, training):
        self.parallel_hidden_state_column_computation.set_training(training)
        self.parallel_memory_state_column_computation.set_training(training)

    def forward(self, x):
        raise NotImplementedError


class MultiDimensionalLSTMParametersCreator:

    # Needs to be implemented in the subclasses
    @abstractmethod
    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels,
                                                               use_dropout: bool):
        raise RuntimeError("not implemented")


class MultiDimensionalLSTMParametersCreatorFast(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                               hidden_states_size, input_channels, use_dropout: bool):
        return MultiDimensionalLSTMParametersOneDirectionFast.\
            create_multi_dimensional_lstm_parameters_one_direction_fast(hidden_states_size, input_channels, use_dropout)


class MultiDimensionalLSTMParametersCreatorSlow(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self, hidden_states_size, input_channels,
                                                               use_dropout: bool):
        return MultiDimensionalLSTMParametersOneDirection.\
            create_multi_dimensional_lstm_parameters_one_direction(hidden_states_size, input_channels, use_dropout)