from modules.multi_dimensional_rnn import MultiDimensionalRNN
from modules.multi_dimensional_rnn import MultiDimensionalRNNBase
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn
import torch.nn as nn
from modules.multi_dimensional_rnn import StateUpdateBlock


class MultiDimensionalLSTM(MultiDimensionalRNNBase):
    def __init__(self, hidden_states_size, batch_size, compute_multi_directional: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalLSTM, self).__init__(hidden_states_size, batch_size,
                                                   compute_multi_directional,
                                                   nonlinearity)
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

        # For multi-directional rnn
        if self.compute_multi_directional:
            self.fc3 = nn.Linear(self.number_of_output_dimensions(), 10)
        else:
            self.fc3 = nn.Linear(self.number_of_output_dimensions(), 10)

        # Set initial bias for the forget gates to one, since it is known to give better results
        self.set_bias_forget_gates_to_one()

        self.state_convolutions = nn.ModuleList([])
        self.register_parameters_to_assure_same_gpu_is_used()

    def set_bias_forget_gates_to_one(self):
        FORGET_GATE_BIAS_INIT = 1
        self.forget_gate_one_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        self.forget_gate_one_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        self.forget_gate_one_memory_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)

        self.forget_gate_two_input_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)
        self.forget_gate_two_hidden_state_update_block.set_bias_for_convolutions(FORGET_GATE_BIAS_INIT)
        self.forget_gate_two_memory_state_convolution.bias.data.fill_(FORGET_GATE_BIAS_INIT)

    def register_parameters_to_assure_same_gpu_is_used(self):
        # This is necessary to make sure things are stored on the same gpu, otherwise
        # pytorch doesn't realizes these convolutions are part of this module
        # See: # https://discuss.pytorch.org/t/runtime-error-tensors-are-on-different-gpus/2100/8
        self.state_convolutions.extend(self.input_hidden_state_update_block.get_state_convolutions_as_list())
        self.state_convolutions.extend(self.input_gate_hidden_state_update_block.get_state_convolutions_as_list())
        self.state_convolutions.extend(self.input_gate_memory_state_update_block.get_state_convolutions_as_list())
        self.state_convolutions.extend(self.forget_gate_one_hidden_state_update_block.get_state_convolutions_as_list())
        self.state_convolutions.extend(self.forget_gate_two_hidden_state_update_block.get_state_convolutions_as_list())
        self.state_convolutions.extend(self.output_gate_hidden_state_update_block.get_state_convolutions_as_list())


    @staticmethod
    def create_multi_dimensional_lstm(hidden_states_size:int ,batch_size:int , compute_multi_directional: bool,
                                     nonlinearity="tanh"):
        return MultiDimensionalLSTM(hidden_states_size, batch_size, compute_multi_directional, nonlinearity)

    def compute_multi_dimensional_lstm_one_direction(self, x):
        # Step 1: Create a skewed version of the input image
        # skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(x)
        skewed_images_variable = MultiDimensionalRNNBase.create_skewed_images_variable_four_dim(x)
        # print("list(x.size()): " + str(list(x.size())))
        image_height = x.size(2)
        # print("image height: " + str(image_height))
        previous_hidden_state_column = Variable(torch.zeros(self.input_channels,
                                                            self.hidden_states_size,
                                                            image_height))
        # Todo: I'm confused about the dimensions of previous_memory_state
        # and previous_hidden_state: why the latter has dimension equal to
        # batch size but for the former it doesn't seem to matter
        previous_memory_state_column = Variable(torch.zeros(self.input_channels,
                                                            self.hidden_states_size,
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
            input_hidden_state_column = self.input_hidden_state_update_block.\
                compute_weighted_states_input(previous_hidden_state_column)

            # print("state_column.size(): " + str(state_column.size()))
            input_state_plus_input = MultiDimensionalRNNBase.compute_states_plus_input(input_input_matrix, column_number,
                                                                                       input_hidden_state_column)

            # Compute the sum of weighted inputs of the input gate
            input_gate_weighted_states_plus_input = self.\
                compute_weighted_input_input_gate(previous_hidden_state_column, previous_memory_state_column,
                                                  column_number, input_gate_input_matrix)


            # Compute the input activation
            input_activation_column = F.tanh(input_state_plus_input)
            # Compute the input gate activation
            input_gate_activation_column = F.sigmoid(input_gate_weighted_states_plus_input)

            input_and_input_gate_combined = torch.mul(input_activation_column, input_gate_activation_column)
            # print("input and input gate combined: " + str(input_and_input_gate_combined))

            memory_states_column_forget_gate_one = StateUpdateBlock.\
                get_previous_state_column_static(previous_memory_state_column, 1,
                                                 self.hidden_states_size)

            forget_gate_one_weighted_stated_plus_input = self.compute_weighted_input_forget_gate(
                self.forget_gate_one_hidden_state_update_block,
                self.forget_gate_one_memory_state_convolution,
                previous_hidden_state_column, previous_memory_state_column,
                column_number, forget_gate_one_input_matrix,
                1)

            # Compute the forget gate one activation
            forget_gate_one_activation_column = F.sigmoid(forget_gate_one_weighted_stated_plus_input)
            # print("forget gate one activation column: " + str(forget_gate_one_activation_column))

            # Compute the activation for forget gate one
            forget_gate_one_activation_multiplied_with_previous_memory_state = \
                torch.mul(forget_gate_one_activation_column,
                          memory_states_column_forget_gate_one)

            memory_states_column_forget_gate_two = StateUpdateBlock.\
                get_previous_state_column_static(previous_memory_state_column, 2,
                                                 self.hidden_states_size)

            forget_gate_two_weighted_stated_plus_input = self.compute_weighted_input_forget_gate(
                self.forget_gate_two_hidden_state_update_block,
                self.forget_gate_two_memory_state_convolution,
                previous_hidden_state_column, previous_memory_state_column,
                column_number, forget_gate_two_input_matrix,
                2)

            # Compute the forget gate two activation
            forget_gate_two_activation_column = F.sigmoid(forget_gate_two_weighted_stated_plus_input)

            # forget_gate_weighted_states_combined =  forget_gate_one_weighted_stated_plus_input + forget_gate_two_weighted_stated_plus_input
            # forget_gates_combined_activation_column = F.sigmoid(forget_gate_weighted_states_combined)
            # forget_gates_combined_activation_multiplied_with_previous_memory_state = torch.mul(
            #    forget_gates_combined_activation_column, previous_memory_state_column)

            # Compute the activation for forget gate two
            forget_gate_two_activation_multiplied_with_previous_memory_state = torch.mul(
                forget_gate_two_activation_column, memory_states_column_forget_gate_two)

            # print("input_and_input_gate_combined: " + str(input_and_input_gate_combined))

            #print("forget_gate_one_activation_multiplied_with_previous_memory_state: "+
            #      str(forget_gate_one_activation_multiplied_with_previous_memory_state))

            #print("forget_gate_one_activation_multiplied_with_previous_memory_state: " +
            #      str(forget_gate_two_activation_multiplied_with_previous_memory_state))

            new_memory_state = input_and_input_gate_combined + \
                forget_gate_two_activation_multiplied_with_previous_memory_state + \
                forget_gate_one_activation_multiplied_with_previous_memory_state # + \
                # forget_gates_combined_activation_multiplied_with_previous_memory_state \


            # print("new memory state: " + str(new_memory_state))

            # This additional tanh activation function taken from the NVIDIA diagram
            # was not in the deep learning book diagram, and does not seem to help
            # really ?
            new_memory_state_activation_column = F.tanh(new_memory_state)

            # Compute the sum of weighted inputs of the ouput gate
            output_gate_weighted_states_plus_input = self. \
                compute_weighted_input_output_gate(previous_hidden_state_column, new_memory_state,
                                                   column_number, output_gate_input_matrix)



            output_gate_activation_column = F.sigmoid(output_gate_weighted_states_plus_input)

            # print("input_column: " + str(input_column))
            #print("state_plus_input: " + str(state_plus_input))
            #activation_column = torch.mul(new_memory_state_activation_column, output_gate_activation_column)
            #activation_column = torch.mul(new_memory_state, output_gate_activation_column)
            #activation_column = self.get_activation_function()(input_state_plus_input)
            activation_column = new_memory_state_activation_column
            # print("output gate activation column: " + str(output_gate_activation_column))
            #print("activation column: " + str(activation_column))

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
                                                hidden_state_update_block,
                                                memory_state_update_block):
        input_gate_hidden_state_column = hidden_state_update_block.\
            compute_weighted_states_input(previous_hidden_state_column)
        input_gate_memory_state_column = memory_state_update_block.\
            compute_weighted_states_input(previous_memory_state_column)
        input_gate_input_column = input_gate_input_matrix[:, :, :, column_number]
        input_gate_weighted_states_plus_weighted_input = input_gate_input_column + input_gate_hidden_state_column + \
            input_gate_memory_state_column
        return input_gate_weighted_states_plus_weighted_input

    def compute_weighted_input_input_gate(self, previous_hidden_state_column, previous_memory_state_column,
                                          column_number, input_gate_input_matrix):
        return self.compute_weighted_input_both_memory_gate(previous_hidden_state_column, previous_memory_state_column,
                                                            column_number, input_gate_input_matrix,
                                                            self.input_gate_hidden_state_update_block,
                                                            self.input_gate_memory_state_update_block)

    def compute_weighted_input_output_gate(self, previous_hidden_state_column, previous_memory_state_column,
                                           column_number, output_gate_input_matrix):
        return self.compute_weighted_input_forget_gate(
                self.output_gate_hidden_state_update_block,
                self.output_gate_memory_state_convolution,
                previous_hidden_state_column, previous_memory_state_column,
                column_number, output_gate_input_matrix,
                2)

    def compute_weighted_input_forget_gate(self, forget_gate_hidden_state_update_block,
                                           forget_gate_memory_state_convolution,
                                           previous_hidden_state_column, previous_memory_state_column,
                                           column_number, forget_gate_input_matrix,
                                           memory_state_index: int):
        forget_gate_hidden_state_column = \
            forget_gate_hidden_state_update_block.compute_weighted_states_input(previous_hidden_state_column)
        forget_gate_memory_state_column = \
            StateUpdateBlock.compute_weighted_state_input_static(forget_gate_memory_state_convolution,
                                                                 previous_memory_state_column,
                                                                 memory_state_index,
                                                                 self.hidden_states_size)
        forget_gate_input_column = forget_gate_input_matrix[:, :, :, column_number]
        forget_gate_weighted_states_plus_weighted_input = forget_gate_input_column + forget_gate_hidden_state_column + \
            forget_gate_memory_state_column

        #print("forget_gate_memory_state_column: " + str(forget_gate_memory_state_column))
        #print("forget_gate_hidden_state_column: " + str(forget_gate_hidden_state_column))
        #print("forget_gate_input_column: " + str(forget_gate_input_column))

        # print("forget_gate_weighted_states_plus_weighted_input: " + str(forget_gate_weighted_states_plus_weighted_input))

        return forget_gate_weighted_states_plus_weighted_input

    def forward_one_directional_multi_dimensional_lstm(self, x):
        activations_unskewed = self.compute_multi_dimensional_lstm_one_direction(x)
        activations_one_dimensional = activations_unskewed.view(-1, self.number_of_output_dimensions())
        # print("activations_one_dimensional: " + str(activations_one_dimensional))
        # It is necessary to output a tensor of size 10, for 10 different output classes
        result = self.fc3(activations_one_dimensional)
        return result

    def _final_activation_function(self, final_activation_function_input):
        return self.fc3(final_activation_function_input)

    # Needs to be implemented in the subclasses
    def _compute_multi_dimensional_function_one_direction(self, function_input):
        return self.compute_multi_dimensional_lstm_one_direction(function_input)

    # Input tensor x is a batch of image tensors
    def forward(self, x):
        if self.compute_multi_directional:
            return self.forward_multi_directional_multi_dimensional_function_fast(x)
        else:
            return self.forward_one_directional_multi_dimensional_lstm(x)
