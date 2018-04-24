from modules.state_update_block import StateUpdateBlock
from modules.parallel_multiple_state_weightings_computation import ParallelMultipleStateWeightingsComputation
from abc import abstractmethod
import torch.nn as nn


class MultiDimensionalLSTMParametersOneDirectionBase:
    def __init__(self, hidden_states_size, input_channels):
        self.input_channels = input_channels
        self.hidden_states_size = hidden_states_size

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
        self.input_gate_memory_state_update_block = StateUpdateBlock(hidden_states_size)
        self.forget_gate_one_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                                  self.hidden_states_size, 1)
        self.forget_gate_two_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                                  self.hidden_states_size, 1)
        self.output_gate_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                              self.hidden_states_size, 1)

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

    def get_all_input_convolutions_as_list(self):
        result = list([])
        result.append(self.input_input_convolution)
        result.append(self.input_gate_input_convolution)
        result.append(self.forget_gate_one_input_convolution)
        result.append(self.forget_gate_two_input_convolution)
        result.append(self.output_gate_input_convolution)
        return result

    def get_all_memory_state_convolutions_as_list(self):
        result = list([])
        result.append(self.output_gate_memory_state_convolution)
        result.append(self.forget_gate_one_memory_state_convolution)
        result.append(self.forget_gate_two_memory_state_convolution)
        result.extend(self.input_gate_memory_state_update_block.get_state_convolutions_as_list())
        return result


class MultiDimensionalLSTMParametersOneDirection(MultiDimensionalLSTMParametersOneDirectionBase):
    def __init__(self, hidden_states_size, input_channels):
        super(MultiDimensionalLSTMParametersOneDirection, self).__init__(hidden_states_size, input_channels)

        self.input_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.input_gate_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.forget_gate_one_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.forget_gate_two_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.output_gate_hidden_state_update_block = StateUpdateBlock(hidden_states_size)

        self.previous_hidden_state_column = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction(hidden_states_size, input_channels):
        return MultiDimensionalLSTMParametersOneDirection(hidden_states_size, input_channels)

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

    def get_all_parameters_as_list(self):
        result = list([])
        result.extend(self.get_all_input_convolutions_as_list())
        result.extend(self.get_all_memory_state_convolutions_as_list())
        result.extend(self.get_all_hidden_state_convolutions_as_list())
        return result


class MultiDimensionalLSTMParametersOneDirectionFast(MultiDimensionalLSTMParametersOneDirectionBase):
    def __init__(self, hidden_states_size, input_channels):
        super(MultiDimensionalLSTMParametersOneDirectionFast, self).__init__(hidden_states_size, input_channels)

        # There are five paired input weightings, in this case, pairs of previous hidden state
        # inputs, namely for: 1) the input, 2) the input gate, 3) forget gate one,
        # 4) forget gate two, 5) the output gate
        self.parallel_hidden_state_column_computation = ParallelMultipleStateWeightingsComputation.create_parallel_multiple_state_weighting_computation(
            hidden_states_size, 5)

        self.node_hidden_state_columns = None
        self.node_memory_state_columns = None  # TODO: Implement the computation of this
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction_fast(hidden_states_size, input_channels):
        return MultiDimensionalLSTMParametersOneDirectionFast(hidden_states_size, input_channels)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        # The hidden state columns for the different computational nodes:
        # 1) the input, 2) the input gate, 3) forget gate one,
        # 4) forget gate two, 5) the output gate
        self.node_hidden_state_columns = \
            self.parallel_hidden_state_column_computation.compute_summed_outputs_every_pair(previous_hidden_state_column)
        self.previous_memory_state_column = previous_memory_state_column

        # TODO: A second list node_memory_state_columns should be computed
        # using a second 1d convolution that computes all memory state convolutions
        # that depend only on the input at once

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
        # TODO: Compute using self.node_memory_state_columns
        input_gate_memory_state_column = self.input_gate_memory_state_update_block. \
            compute_weighted_states_input(self.previous_memory_state_column)
        return input_gate_memory_state_column

    def get_forget_gate_one_memory_state_column(self):
        # TODO: Compute using self.node_memory_state_columns
        forget_gate_memory_state_column = \
            StateUpdateBlock.compute_weighted_state_input_state_one(self.forget_gate_one_memory_state_convolution,
                                                                    self.previous_memory_state_column)
        return forget_gate_memory_state_column

    def get_forget_gate_two_memory_state_column(self):
        # TODO: Compute using self.node_memory_state_columns
        forget_gate_memory_state_column = \
            StateUpdateBlock.compute_weighted_state_input_state_two(self.forget_gate_two_memory_state_convolution,
                                                                    self.previous_memory_state_column)
        return forget_gate_memory_state_column

    def get_all_parameters_as_list(self):
        result = list([])
        result.extend(self.get_all_input_convolutions_as_list())
        result.extend(self.get_all_memory_state_convolutions_as_list())
        result.extend(self.parallel_hidden_state_column_computation.get_state_convolutions_as_list())
        return result


class MultiDimensionalLSTMParametersCreator:

    # Needs to be implemented in the subclasses
    @abstractmethod
    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                                    hidden_states_size, input_channels):
        raise RuntimeError("not implemented")


class MultiDimensionalLSTMParametersCreatorFast(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                                    hidden_states_size, input_channels):
        return MultiDimensionalLSTMParametersOneDirectionFast.\
            create_multi_dimensional_lstm_parameters_one_direction_fast(hidden_states_size, input_channels)


class MultiDimensionalLSTMParametersCreatorSlow(MultiDimensionalLSTMParametersCreator):

    def create_multi_dimensional_lstm_parameters_one_direction(self,
                                                                    hidden_states_size, input_channels):
        return MultiDimensionalLSTMParametersOneDirectionFast.\
            create_multi_dimensional_lstm_parameters_one_direction(hidden_states_size, input_channels)