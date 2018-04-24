from modules.state_update_block import StateUpdateBlock
import torch.nn as nn


class MultiDimensionalLSTMParametersOneDirection():
    def __init__(self, hidden_states_size, input_channels):
        self.input_channels = input_channels
        self.hidden_states_size = hidden_states_size

        # Input
        self.input_input_convolution = nn.Conv2d(self.input_channels,
                                                 self.hidden_states_size, 1)

        self.input_hidden_state_update_block = StateUpdateBlock(hidden_states_size)

        # Input gate
        self.input_gate_input_convolution = nn.Conv2d(self.input_channels,
                                                      self.hidden_states_size, 1)

        self.input_gate_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.input_gate_memory_state_update_block = StateUpdateBlock(hidden_states_size)

        # Forget gate 1
        self.forget_gate_one_input_convolution = nn.Conv2d(self.input_channels,
                                                           self.hidden_states_size, 1)
        self.forget_gate_one_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        self.forget_gate_one_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                                  self.hidden_states_size, 1)

        # Forget gate 2
        self.forget_gate_two_input_convolution = nn.Conv2d(self.input_channels,
                                                           self.hidden_states_size, 1)
        self.forget_gate_two_hidden_state_update_block = StateUpdateBlock(hidden_states_size)

        #self.forget_gate_two_hidden_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        self.forget_gate_two_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                                  self.hidden_states_size, 1)

        # Output gate
        self.output_gate_input_convolution = nn.Conv2d(self.input_channels,
                                                       self.hidden_states_size, 1)

        self.output_gate_hidden_state_update_block = StateUpdateBlock(hidden_states_size)
        # self.output_gate_memory_state_update_block = StateUpdateBlock(hidden_states_size)
        self.output_gate_memory_state_convolution = nn.Conv1d(self.hidden_states_size,
                                                                  self.hidden_states_size, 1)

        self.previous_hidden_state_column = None
        self.previous_memory_state_column = None

    @staticmethod
    def create_multi_dimensional_lstm_parameters_one_direction(hidden_states_size, input_channels):
        return MultiDimensionalLSTMParametersOneDirection(hidden_states_size, input_channels)

    def prepare_computation_next_column_functions(self, previous_hidden_state_column,
                                                  previous_memory_state_column):
        self.previous_hidden_state_column = previous_hidden_state_column
        self.previous_memory_state_column = previous_memory_state_column

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

    def get_output_gate_hidden_state_column(self):
        output_gate_hidden_state_column = \
            self.output_gate_hidden_state_update_block.compute_weighted_states_input(
                self.previous_hidden_state_column)
        return output_gate_hidden_state_column

    def get_all_parameters_as_list(self):
        result = list([])
        result.append(self.input_input_convolution)
        result.append(self.input_gate_input_convolution)
        result.append(self.forget_gate_one_input_convolution)
        result.append(self.forget_gate_two_input_convolution)
        result.append(self.output_gate_input_convolution)
        result.append(self.output_gate_memory_state_convolution)
        result.append(self.forget_gate_one_memory_state_convolution)
        result.append(self.forget_gate_two_memory_state_convolution)
        result.extend(self.input_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.input_gate_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.input_gate_memory_state_update_block.get_state_convolutions_as_list())
        result.extend(self.forget_gate_one_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.forget_gate_two_hidden_state_update_block.get_state_convolutions_as_list())
        result.extend(self.output_gate_hidden_state_update_block.get_state_convolutions_as_list())

        return result
