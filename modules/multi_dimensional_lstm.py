from modules.multi_dimensional_rnn import MultiDimensionalRNN
from modules.multi_dimensional_rnn import MultiDimensionalRNNBase
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn
import torch.nn as nn


class MultiDimensionalLSTM(MultiDimensionalRNNBase):
    def __init__(self, batch_size, compute_multi_directional: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalLSTM, self).__init__(batch_size,
                                                  compute_multi_directional,
                                                  nonlinearity)
        # Input
        self.input_input_convolution = nn.Conv2d(self.input_channels,
                                                 self.output_channels, 1)
        self.input_hidden_state_convolution = nn.Conv1d(self.input_channels,
                                                        self.output_channels, 2,
                                                        padding=1)

        # Input gate
        self.input_gate_input_convolution = nn.Conv2d(self.input_channels,
                                                      self.output_channels, 1)
        self.input_gate_hidden_state_convolution = nn.Conv1d(self.input_channels,
                                                             self.output_channels, 2,
                                                             padding=1)
        self.input_gate_memory_state_convolution = nn.Conv1d(self.input_channels,
                                                             self.output_channels, 2,
                                                             padding=1)

        # Forget gate 1
        self.forget_gate_one_input_convolution = nn.Conv2d(self.input_channels,
                                                           self.output_channels, 1)
        self.forget_gate_one_hidden_state_convolution = nn.Conv1d(self.input_channels,
                                                                  self.output_channels, 2,
                                                                  padding=1)
        self.forget_gate_one_memory_state_convolution = nn.Conv1d(self.input_channels,
                                                                  self.output_channels, 1)

        # Forget gate 2
        self.forget_gate_two_input_convolution = nn.Conv2d(self.input_channels,
                                                           self.output_channels, 1)
        self.forget_gate_two_hidden_state_convolution = nn.Conv1d(self.input_channels,
                                                                  self.output_channels, 2,
                                                                  padding=1)
        self.forget_gate_two_memory_state_convolution = nn.Conv1d(self.input_channels,
                                                                  self.output_channels, 1)

        # Output gate
        self.output_gate_input_convolution = nn.Conv2d(self.input_channels,
                                                           self.output_channels, 1)
        self.output_gate_hidden_state_convolution = nn.Conv1d(self.input_channels,
                                                                  self.output_channels, 2,
                                                                  padding=1)
        self.output_gate_memory_state_convolution = nn.Conv1d(self.input_channels,
                                                                  self.output_channels, 2,
                                                                  padding = 1)


        # self.fc3 = nn.Linear(1024, 10)
        # For multi-directional rnn
        if self.compute_multi_directional:
            self.fc3 = nn.Linear(1024 * 4, 10)
        else:
            self.fc3 = nn.Linear(1024, 10)

    @staticmethod
    def create_multi_dimensional_lstm(batch_size, compute_multi_directional: bool,
                                     nonlinearity="tanh"):
        return MultiDimensionalLSTM(batch_size, compute_multi_directional, nonlinearity)

    def compute_multi_dimensional_lstm_one_direction(self, x):
        # Step 1: Create a skewed version of the input image
        # skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(x)
        skewed_images_variable = MultiDimensionalRNNBase.create_skewed_images_variable_four_dim(x)
        # print("list(x.size()): " + str(list(x.size())))
        image_height = x.size(2)
        # print("image height: " + str(image_height))
        previous_hidden_state_column = Variable(torch.zeros(self.input_channels,
                                                            self.output_channels,
                                                            image_height))
        # Todo: I'm confused about the dimensions of previous_memory_state
        # and previous_hidden_state: why the latter has dimension equal to
        # batch size but for the former it doesn't seem to matter
        previous_memory_state_column = Variable(torch.ones(self.input_channels,
                                                           self.output_channels,
                                                           image_height))

        if MultiDimensionalRNNBase.use_cuda():
            previous_hidden_state_column = previous_hidden_state_column.cuda()
            previous_memory_state_column = previous_memory_state_column.cuda()

        skewed_image_columns = skewed_images_variable.size(3)

        input_input_matrix = self.input_input_convolution(skewed_images_variable)
        # print("input_matrix: " + str(input_matrix))

        input_gate_input_matrix = self.input_gate_input_convolution(skewed_images_variable)
        forget_gate_one_input_matrix = self.forget_gate_one_input_convolution(skewed_images_variable)
        forget_gate_two_input_matrix = self.forget_gate_two_input_convolution(skewed_images_variable)
        output_gate_input_matrix = self.output_gate_input_convolution(skewed_images_variable)

        activations = list([])

        # print("skewed image columns: " + str(skewed_image_columns))

        for column_number in range(0, skewed_image_columns):
            # Compute convolution on previous state column vector padded with zeros
            # Compute convolution on previous state column vector padded with zeros
            input_hidden_state_column = MultiDimensionalRNNBase.compute_state_convolution_and_remove_bottom_padding(
                self.input_hidden_state_convolution, previous_hidden_state_column, image_height)

            # print("state_column.size(): " + str(state_column.size()))
            input_state_plus_input = MultiDimensionalRNNBase.compute_state_plus_input(input_input_matrix, column_number,
                                                                                      input_hidden_state_column)

            # Compute the sum of weighted inputs of the input gate
            input_gate_weighted_states_plus_input = self.\
                compute_weighted_input_input_gate(previous_hidden_state_column, previous_memory_state_column,
                                                  column_number, input_gate_input_matrix,
                                                  image_height)




            # Compute the input activation
            input_activation_column = F.tanh(input_state_plus_input)
            # Compute the input gate activation
            input_gate_activation_column = F.sigmoid(input_gate_weighted_states_plus_input)

            input_and_input_gate_combined = torch.mul(input_gate_activation_column, input_gate_activation_column)
            # print("input and input gate combined: " + str(input_and_input_gate_combined))

            memory_states_column_forget_gate_one = self.get_previous_memory_state_column_input_forget_gate(
                previous_memory_state_column, 1)

            forget_gate_one_weighted_stated_plus_input = self.compute_weighted_input_forget_gate(
                self.forget_gate_one_hidden_state_convolution,
                self.forget_gate_one_memory_state_convolution,
                previous_hidden_state_column, previous_memory_state_column,
                column_number, forget_gate_one_input_matrix,
                1,
                image_height)

            # Compute the forget gate one activation
            forget_gate_one_activation_column = F.sigmoid(forget_gate_one_weighted_stated_plus_input)
            # print("forget gate one activation column: " + str(forget_gate_one_activation_column))

            # Compute the activation for forget gate one
            forget_gate_one_activation_multiplied_with_previous_memory_state = torch.mul(forget_gate_one_activation_column,
                                                   memory_states_column_forget_gate_one)

            memory_states_column_forget_gate_two = self.get_previous_memory_state_column_input_forget_gate(
                previous_memory_state_column, 2)

            forget_gate_two_weighted_stated_plus_input = self.compute_weighted_input_forget_gate(
                self.forget_gate_two_hidden_state_convolution,
                self.forget_gate_two_memory_state_convolution,
                previous_hidden_state_column, previous_memory_state_column,
                column_number, forget_gate_two_input_matrix,
                2,
                image_height)

            # Compute the forget gate two activation
            forget_gate_two_activation_column = F.sigmoid(forget_gate_two_weighted_stated_plus_input)

            # Compute the activation for forget gate two
            forget_gate_two_activation_multiplied_with_previous_memory_state = torch.mul(
                forget_gate_two_activation_column, memory_states_column_forget_gate_two)

            new_memory_state = input_and_input_gate_combined + \
                forget_gate_one_activation_multiplied_with_previous_memory_state + \
                forget_gate_two_activation_multiplied_with_previous_memory_state

            new_memory_state_activation_column = F.tanh(new_memory_state)

            ### TODO Implement output gate etc
            # Compute the sum of weighted inputs of the ouput gate
            output_gate_weighted_states_plus_input = self. \
                compute_weighted_input_output_gate(previous_hidden_state_column, new_memory_state,
                                                  column_number, output_gate_input_matrix,
                                                  image_height)

            output_gate_activation_column = F.sigmoid(output_gate_weighted_states_plus_input)

            # print("input_column: " + str(input_column))
            # print("state_plus_input: " + str(state_plus_input))
            activation_column = torch.mul(new_memory_state_activation_column, output_gate_activation_column)
            #activation_column = self.get_activation_function()(input_state_plus_input)
            # print("activation column: " + str(activation_column))

            previous_hidden_state_column = activation_column
            previous_memory_state_column = new_memory_state
            activations.append(activation_column)

        original_image_columns = x.size(2)
        skewed_image_rows = skewed_images_variable.size(2)
        activations_unskewed = MultiDimensionalRNNBase.extract_unskewed_activations(activations,
                                                                                    original_image_columns,
                                                                                    skewed_image_columns,
                                                                                            skewed_image_rows)
        # print("activations_unskewed: " + str(activations_unskewed))
        return activations_unskewed

    def compute_weighted_input_both_memory_gate(self, previous_hidden_state_column, previous_memory_state_column,
                                          column_number, input_gate_input_matrix,
                                          hidden_state_convolution,
                                          memory_state_convolution,
                                          image_height):
        input_gate_hidden_state_column = MultiDimensionalRNNBase. \
            compute_state_convolution_and_remove_bottom_padding(hidden_state_convolution,
                                                                previous_hidden_state_column,
                                                                image_height)
        input_gate_memory_state_column = MultiDimensionalRNNBase. \
            compute_state_convolution_and_remove_bottom_padding(memory_state_convolution,
                                                                previous_memory_state_column,
                                                                image_height)
        input_gate_input_column = input_gate_input_matrix[:, :, :, column_number]
        input_gate_weighted_states_plus_weighted_input = input_gate_input_column + input_gate_hidden_state_column + \
            input_gate_memory_state_column
        return input_gate_weighted_states_plus_weighted_input

    def compute_weighted_input_input_gate(self, previous_hidden_state_column, previous_memory_state_column,
                                          column_number, input_gate_input_matrix,
                                          image_height):
        return self.compute_weighted_input_both_memory_gate(previous_hidden_state_column, previous_memory_state_column,
                                          column_number, input_gate_input_matrix,
                                          self.input_hidden_state_convolution,
                                          self.input_gate_memory_state_convolution,
                                          image_height)

    def compute_weighted_input_output_gate(self, previous_hidden_state_column, previous_memory_state_column,
                                          column_number, input_gate_input_matrix,
                                          image_height):
        return self.compute_weighted_input_both_memory_gate(previous_hidden_state_column, previous_memory_state_column,
                                          column_number, input_gate_input_matrix,
                                          self.output_gate_hidden_state_convolution,
                                          self.output_gate_memory_state_convolution,
                                          image_height)



    def compute_weighted_input_forget_gate(self, forget_gate_hidden_state_convolution,
                                           forget_gate_memory_state_convolution,
                                           previous_hidden_state_column, previous_memory_state_column,
                                           column_number, forget_gate_input_matrix,
                                           memory_state_index: int,
                                           image_height):
        forget_gate_hidden_state_column = MultiDimensionalRNNBase. \
            compute_state_convolution_and_remove_bottom_padding(forget_gate_hidden_state_convolution,
                                                                previous_hidden_state_column,
                                                                image_height)
        forget_gate_memory_state_column = forget_gate_memory_state_convolution(
            self.get_previous_memory_state_column_input_forget_gate(previous_memory_state_column,
                                                                    memory_state_index))
        forget_gate_input_column = forget_gate_input_matrix[:, :, :, column_number]
        forget_gate_weighted_states_plus_weighted_input = forget_gate_input_column + forget_gate_hidden_state_column + \
            forget_gate_memory_state_column
        return forget_gate_weighted_states_plus_weighted_input

    def get_previous_memory_state_column_input_forget_gate(self, previous_memory_state_column,
                                                           memory_state_index: int):
        #print("previous memory state column: " + str(previous_memory_state_column))
        if memory_state_index == 1:
            previous_memory_state_column_shifted = previous_memory_state_column.clone()
            height = previous_memory_state_column.size(2)
            zeros_padding = Variable(torch.zeros(previous_memory_state_column.size(0), 1, 1))
            if self.use_cuda():
                zeros_padding = zeros_padding.cuda()
            skip_first_sub_tensor = previous_memory_state_column_shifted[:, :, 0:(height - 1)]
            # print("zeros padding" + str(zeros_padding))
            # print("skip_first_sub_tensor: " + str(skip_first_sub_tensor))
            previous_memory_state_column_shifted = torch.\
                cat((zeros_padding, skip_first_sub_tensor), 2)
            # print("Returning previous_memory_state_column_shifted: " + str(previous_memory_state_column_shifted))
            return previous_memory_state_column_shifted
        return previous_memory_state_column

        return previous_memory_state_column

    def forward_one_directional_multi_dimensional_lstm(self, x):
        activations_unskewed = self.compute_multi_dimensional_lstm_one_direction(x)
        activations_one_dimensional = activations_unskewed.view(-1, 1024)
        # print("activations_one_dimensional: " + str(activations_one_dimensional))
        # It is nescessary to output a tensor of size 10, for 10 different output classes
        result = self.fc3(activations_one_dimensional)
        return result

    # Input tensor x is a batch of image tensors
    def forward(self, x):
        return self.forward_one_directional_multi_dimensional_lstm(x)
