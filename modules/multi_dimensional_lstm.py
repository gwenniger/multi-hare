from modules.multi_dimensional_rnn import MultiDimensionalRNN
from modules.multi_dimensional_rnn import MultiDimensionalRNNBase
import util.tensor_flipping
import torch
import torch.nn.functional as F
import torch.nn
import torch.nn as nn
from modules.state_update_block import StateUpdateBlock
from modules.multi_dimensional_lstm_parameters import MultiDimensionalLSTMParametersOneDirection
from modules.multi_dimensional_lstm_parameters import MultiDimensionalLSTMParametersOneDirectionFast
from modules.multi_dimensional_lstm_parameters import MultiDimensionalLSTMParametersCreator
from modules.multi_dimensional_lstm_parameters import MultiDimensionalLSTMParametersCreatorSlow
from modules.multi_dimensional_lstm_parameters import MultiDimensionalLSTMParametersCreatorFast
from util.image_input_transformer import ImageInputTransformer
from modules.inside_model_gradient_clipping import InsideModelGradientClamping


def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())
    print('grad_output norm: ', grad_output[0].norm())


class MultiDimensionalLSTM(MultiDimensionalRNNBase):

    def __init__(self, input_channels: int, hidden_states_size: int, compute_multi_directional: bool,
                 clamp_gradients: bool,
                 use_dropout: bool, training: bool,
                 multi_dimensional_lstm_parameter_creator:MultiDimensionalLSTMParametersCreator,
                 nonlinearity="tanh"):
        super(MultiDimensionalLSTM, self).__init__(input_channels, hidden_states_size, compute_multi_directional,
                                                   nonlinearity)

        self.clamp_gradients = clamp_gradients
        if self.clamp_gradients:
            print("MultiDimensionalLSTM - clamp_gradients=" + str(self.clamp_gradients))
        else:
            print("WARNING: MultiDimensionalLSTM - clamp_gradients=" + str(self.clamp_gradients))

        self.use_dropout = use_dropout
        self.training = training

        self.mdlstm_direction_one_parameters = \
            multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
                self.hidden_states_size, self.input_channels, self.clamp_gradients, use_dropout)

        # Set initial bias for the forget gates to one, since it is known to give better results
        self.mdlstm_direction_one_parameters.set_bias_forget_gates_to_one()

        # For multi-directional rnn
        if self.compute_multi_directional_flag:
            self.mdlstm_direction_two_parameters = \
                multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
                    self.hidden_states_size, self.input_channels, self.clamp_gradients, use_dropout)
            # Set initial bias for the forget gates to one, since it is known to give better results
            self.mdlstm_direction_two_parameters.set_bias_forget_gates_to_one()

            self.mdlstm_direction_three_parameters = \
                multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
                    self.hidden_states_size, self.input_channels, self.clamp_gradients, use_dropout)
            # Set initial bias for the forget gates to one, since it is known to give better results
            self.mdlstm_direction_three_parameters.set_bias_forget_gates_to_one()

            self.mdlstm_direction_four_parameters = \
                multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
                    self.hidden_states_size, self.input_channels, self.clamp_gradients, use_dropout)
            # Set initial bias for the forget gates to one, since it is known to give better results
            self.mdlstm_direction_four_parameters.set_bias_forget_gates_to_one()

        self.state_convolutions = nn.ModuleList([])
        self.register_parameters_to_assure_same_gpu_is_used()

        # See: https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
        # self.register_backward_hook(printgradnorm)

    def register_parameters_to_assure_same_gpu_is_used(self):
        self.state_convolutions.extend(self.mdlstm_direction_one_parameters.get_all_parameters_as_list())

        if self.compute_multi_directional_flag:
            self.state_convolutions.extend(self.mdlstm_direction_two_parameters.get_all_parameters_as_list())
            self.state_convolutions.extend(self.mdlstm_direction_three_parameters.get_all_parameters_as_list())
            self.state_convolutions.extend(self.mdlstm_direction_four_parameters.get_all_parameters_as_list())

    @staticmethod
    def create_multi_dimensional_lstm(input_channels: int, hidden_states_size: int, compute_multi_directional: bool,
                                      clamp_gradients: bool,
                                      use_dropout: bool,
                                      nonlinearity="tanh"):
        return MultiDimensionalLSTM(input_channels, hidden_states_size, compute_multi_directional,
                                    clamp_gradients, use_dropout,
                                    True,
                                    MultiDimensionalLSTMParametersCreatorSlow(),
                                    nonlinearity)

    @staticmethod
    def create_multi_dimensional_lstm_fast(input_channels: int, hidden_states_size: int,
                                           compute_multi_directional: bool,
                                           clamp_gradients: bool,
                                           use_dropout: bool,
                                           nonlinearity="tanh"):
        return MultiDimensionalLSTM(input_channels, hidden_states_size, compute_multi_directional,
                                    clamp_gradients, use_dropout,
                                    True,
                                    MultiDimensionalLSTMParametersCreatorFast(),
                                    nonlinearity)

    def set_training(self, training):
        self.mdlstm_direction_one_parameters.set_training(training)

        if self.compute_multi_directional_flag:
            self.mdlstm_direction_two_parameters.set_training(training)
            self.mdlstm_direction_three_parameters.set_training(training)
            self.mdlstm_direction_four_parameters.set_training(training)

        self.training = training

    def compute_multi_dimensional_lstm_one_direction(self, mdlstm_parameters, x):
        if MultiDimensionalRNNBase.use_cuda():
            # https://discuss.pytorch.org/t/which-device-is-model-tensor-stored-on/4908/7
            device = x.get_device()

        # print("compute_multi_dimensional_lstm_one_direction - x.size(): " + str(x.size()))
        # print("compute_multi_dimensional_lstm_one_direction - self.hidden_states_size: " + str(self.hidden_states_size))

        # Step 1: Create a skewed version of the input image
        # skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(x)
        skewed_images_variable = ImageInputTransformer.create_skewed_images_variable_four_dim(x)
        # print("list(x.size()): " + str(list(x.size())))
        image_height = x.size(2)
        number_of_images = x.size(0)
        # print("image height: " + str(image_height))
        previous_hidden_state_column = torch.zeros(number_of_images,
                                                   self.hidden_states_size,
                                                   image_height)

        # and previous_hidden_state: why the latter has dimension equal to
        # batch size but for the former it doesn't seem to matter
        previous_memory_state_column = torch.zeros(number_of_images,
                                                   self.hidden_states_size,
                                                   image_height)

        # After initialization, the value of grad_fn is still None, later it gets set
        # print("initialization: previous_memory_state_column.grad_fn: " + str(previous_memory_state_column.grad_fn))
        # print("initialization: previous_hidden_state_column.grad_fn: " + str(previous_hidden_state_column.grad_fn))

        if MultiDimensionalRNNBase.use_cuda():
            previous_hidden_state_column = previous_hidden_state_column.to(device)
            previous_memory_state_column = previous_memory_state_column.to(device)

        skewed_image_columns = skewed_images_variable.size(3)

        # print("mdlstm_parameters.input_input_convolution: " + str(mdlstm_parameters.input_input_convolution))
        # print("skewed_images_variable.get_device(): " + str(skewed_images_variable.get_device()))
        # print("mdlstm_parameters.input_input_convolution.bias: "
        # + str(mdlstm_parameters.input_input_convolution.bias))

        mdlstm_parameters.prepare_input_convolutions(skewed_images_variable)

        # Compute the different input convolutions
        input_input_matrix = mdlstm_parameters.get_input_input_matrix()
        # print("input_input_matrix.size(): " + str(input_input_matrix.size()))
        input_gate_input_matrix = mdlstm_parameters.get_input_gate_input_matrix()
        forget_gate_one_input_matrix = mdlstm_parameters.get_forget_gate_one_input_matrix()
        forget_gate_two_input_matrix = mdlstm_parameters.get_forget_gate_two_input_matrix()
        output_gate_input_matrix = mdlstm_parameters.get_output_gate_input_matrix()
        # Cleanup the temporarily stored results, so that they won't be held in
        # memory unnecessarily
        mdlstm_parameters.cleanup_input_convolution_results()

        # if self.clamp_gradients:
        #     # print("MultiDimensionalLSTM.compute_multi_dimensional_lstm_one_direction - register gradient clamping...")
        #     input_input_matrix = InsideModelGradientClamping.register_gradient_clamping(input_input_matrix)
        #     input_gate_input_matrix = InsideModelGradientClamping.register_gradient_clamping(input_gate_input_matrix)
        #     forget_gate_one_input_matrix = InsideModelGradientClamping.register_gradient_clamping(forget_gate_one_input_matrix)
        #     forget_gate_two_input_matrix = InsideModelGradientClamping.register_gradient_clamping(forget_gate_two_input_matrix)
        #     output_gate_input_matrix = InsideModelGradientClamping.register_gradient_clamping(output_gate_input_matrix)

        activations = list([])

        # print("skewed image columns: " + str(skewed_image_columns))

        for column_number in range(0, skewed_image_columns):
            #print("column_number: " + str(column_number))
            #print("previous_hidden_state_column.is_leaf: " + str(previous_hidden_state_column.is_leaf))
            #print("previous_hidden_state_column.grad_fn: " + str(previous_hidden_state_column.grad_fn))
            #print("previous_memory_state_column.is_leaf: " + str(previous_memory_state_column.is_leaf))
            #print("previous_memory_state_column.grad_fn: " + str(previous_memory_state_column.grad_fn))

            #Preparation of the computations of the next state. This involves either just
            # storing the previous hidden state and previous memory state columns in the
            # mdlstm_parameters class or already part of the computation, depending on the
            # implementaiton of mdlstm_parameters

            # print("previous hidden state column: " + str(previous_hidden_state_column))
            # print("previous memory state column: " + str(previous_memory_state_column))
            mdlstm_parameters.prepare_computation_next_column_functions(previous_hidden_state_column,
                                                                        previous_memory_state_column)


            # Compute convolution on previous state column vector padded with zeros
            # Compute convolution on previous state column vector padded with zeros
            input_hidden_state_column = mdlstm_parameters.get_input_hidden_state_column()

            # print("state_column.size(): " + str(state_column.size()))
            input_state_plus_input = MultiDimensionalRNNBase.compute_states_plus_input(input_input_matrix,
                                                                                       column_number,
                                                                                       input_hidden_state_column)

            # Compute the sum of weighted inputs of the input gate
            input_gate_weighted_states_plus_input = MultiDimensionalLSTM.\
                compute_weighted_input_input_gate(column_number, input_gate_input_matrix,
                                                  mdlstm_parameters)

            # Clamp before activation functions
            if self.clamp_gradients:
                input_gate_weighted_states_plus_input = \
                     InsideModelGradientClamping.register_gradient_clamping(input_gate_weighted_states_plus_input)
                input_state_plus_input = InsideModelGradientClamping.register_gradient_clamping(input_state_plus_input)

            # Compute the input activation
            input_activation_column = F.tanh(input_state_plus_input)
            #input_activation_column = F.relu(input_state_plus_input) # Relu can be used as an alternative to tanh
            # Compute the input gate activation
            input_gate_activation_column = F.sigmoid(input_gate_weighted_states_plus_input)

            input_and_input_gate_combined = torch.mul(input_activation_column, input_gate_activation_column)

            # if self.clamp_gradients:
            #     input_and_input_gate_combined = \
            #         InsideModelGradientClamping.register_gradient_clamping(input_and_input_gate_combined)


            # print("input and input gate combined: " + str(input_and_input_gate_combined))

            memory_states_column_forget_gate_one = previous_memory_state_column

            forget_gate_one_weighted_states_plus_input = self.compute_weighted_input_forget_gate(
                mdlstm_parameters.get_forget_gate_one_hidden_state_column(),
                mdlstm_parameters.get_forget_gate_one_memory_state_column(),
                column_number, forget_gate_one_input_matrix)

            if self.clamp_gradients:
                forget_gate_one_weighted_states_plus_input = \
                    InsideModelGradientClamping.register_gradient_clamping(forget_gate_one_weighted_states_plus_input)


            # print(">>> forget_gate_one_weighted_states_plus_input: " + str(forget_gate_one_weighted_states_plus_input))

            # Compute the forget gate one activation
            forget_gate_one_activation_column = F.sigmoid(forget_gate_one_weighted_states_plus_input)
            # print("forget gate one activation column: " + str(forget_gate_one_activation_column))

            # Compute the activation for forget gate one
            forget_gate_one_activation_multiplied_with_previous_memory_state = \
                torch.mul(forget_gate_one_activation_column,
                          memory_states_column_forget_gate_one)

            # if self.clamp_gradients:
            #     forget_gate_one_activation_multiplied_with_previous_memory_state = \
            #         InsideModelGradientClamping.register_gradient_clamping(
            #             forget_gate_one_activation_multiplied_with_previous_memory_state)

            memory_states_column_forget_gate_two = StateUpdateBlock.\
                get_shifted_column_fast(previous_memory_state_column)

            forget_gate_two_weighted_states_plus_input = self.compute_weighted_input_forget_gate(
                mdlstm_parameters.get_forget_gate_two_hidden_state_column(),
                mdlstm_parameters.get_forget_gate_two_memory_state_column(),
                column_number, forget_gate_two_input_matrix)

            if self.clamp_gradients:
                forget_gate_two_weighted_states_plus_input = \
                    InsideModelGradientClamping.register_gradient_clamping(
                        forget_gate_two_weighted_states_plus_input)

            # Compute the forget gate two activation
            forget_gate_two_activation_column = F.sigmoid(forget_gate_two_weighted_states_plus_input)


            # forget_gate_weighted_states_combined =  forget_gate_one_weighted_stated_plus_input + forget_gate_two_weighted_stated_plus_input
            # forget_gates_combined_activation_column = F.sigmoid(forget_gate_weighted_states_combined)
            # forget_gates_combined_activation_multiplied_with_previous_memory_state = torch.mul(
            #    forget_gates_combined_activation_column, previous_memory_state_column)

            # Compute the activation for forget gate two
            forget_gate_two_activation_multiplied_with_previous_memory_state = torch.mul(
                forget_gate_two_activation_column, memory_states_column_forget_gate_two)

            # if self.clamp_gradients:
            #     forget_gate_two_activation_multiplied_with_previous_memory_state = \
            #         InsideModelGradientClamping.register_gradient_clamping(
            #             forget_gate_two_activation_multiplied_with_previous_memory_state)

            # print("input_and_input_gate_combined: " + str(input_and_input_gate_combined))

            # print("forget_gate_one_activation_column: " + str(forget_gate_two_activation_column))
            # print("memory_states_column_forget_gate_one: " + str(memory_states_column_forget_gate_one))
            # print("forget_gate_two_activation_column: " + str(forget_gate_two_activation_column))
            #print("memory_states_column_forget_gate_two: " + str(memory_states_column_forget_gate_two))
            #print("forget_gate_one_activation_multiplied_with_previous_memory_state: "+
            #      str(forget_gate_one_activation_multiplied_with_previous_memory_state))
            #print("forget_gate_two_activation_multiplied_with_previous_memory_state: " +
            #      str(forget_gate_two_activation_multiplied_with_previous_memory_state))

            new_memory_state = input_and_input_gate_combined + \
                forget_gate_two_activation_multiplied_with_previous_memory_state + \
                forget_gate_one_activation_multiplied_with_previous_memory_state # + \
                # forget_gates_combined_activation_multiplied_with_previous_memory_state \

            if self.clamp_gradients:
                new_memory_state = \
                    InsideModelGradientClamping.register_gradient_clamping(new_memory_state)

            #new_memory_state = input_and_input_gate_combined + \
            #    forget_gate_two_activation_multiplied_with_previous_memory_state


            # print("new memory state: " + str(new_memory_state))

            # This additional tanh activation function taken from the NVIDIA diagram
            # was not in the deep learning book diagram, and does not seem to help
            # really ?
            # new_memory_state_activation_column = F.tanh(new_memory_state)

            # Compute the sum of weighted inputs of the ouput gate
            output_gate_weighted_states_plus_input = self. \
                compute_weighted_input_output_gate(mdlstm_parameters, new_memory_state,
                                                   column_number, output_gate_input_matrix)

            if self.clamp_gradients:
                output_gate_weighted_states_plus_input = \
                    InsideModelGradientClamping.register_gradient_clamping(output_gate_weighted_states_plus_input)

            output_gate_activation_column = F.sigmoid(output_gate_weighted_states_plus_input)

            # print("input_column: " + str(input_column))
            #print("state_plus_input: " + str(state_plus_input))

            # This is according to the NVIDIA LSTM diagram
            # activation_column = torch.mul(new_memory_state_activation_column, output_gate_activation_column)

            # This is following the deep learning book
            activation_column = torch.mul(new_memory_state, output_gate_activation_column)

            #activation_column = self.get_activation_function()(input_state_plus_input)
            # activation_column = new_memory_state_activation_column
            # print("output gate activation column: " + str(output_gate_activation_column))
            #print("activation column: " + str(activation_column))

            previous_hidden_state_column = activation_column
            previous_memory_state_column = new_memory_state
            activations.append(activation_column)

            # In the loop the value of grad_fn becomes set, as a backwards path for
            # back-propagation is collected
            # print("in loop: previous_memory_state_column.grad_fn: " + str(previous_memory_state_column.grad_fn))
            # print("in loop: previous_hidden_state_column.grad_fn: " + str(previous_hidden_state_column.grad_fn))

        # print(">>> x.size(): " + str(x.size()))
        original_image_columns = x.size(3)
        skewed_image_rows = skewed_images_variable.size(2)

        activations_unskewed = ImageInputTransformer.extract_unskewed_activations_from_activation_columns(activations,
                                                                                                            original_image_columns,
                                                                                                            skewed_image_columns,
                                                                                                            skewed_image_rows)

        # print("activations_unskewed: " + str(activations_unskewed))
        return activations_unskewed

        # This function is slow because all four function calls for 4 directions are
        # executed sequentially. It isn't entirely clear how to optimize this.
        # See the discussion at:
        # https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360

    def forward_multi_directional_multi_dimensional_lstm(self, x):
        # print("list(x.size()): " + str(list(x.size())))

        # Original order
        activations_unskewed_direction_one = self.\
            compute_multi_dimensional_lstm_one_direction(self.mdlstm_direction_one_parameters, x)

        # Flipping 2nd dimension
        height_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, False)
        activations_unskewed_direction_two_flipped = self.compute_multi_dimensional_lstm_one_direction(
            self.mdlstm_direction_two_parameters, height_flipping.flip(x))
        # Flip back the activations to get the retrieve the original orientation
        activations_unskewed_direction_two = height_flipping.flip(activations_unskewed_direction_two_flipped)

        # print("activations_one_dimensional_two: " + str(activations_one_dimensional_two))

        # Flipping 3th dimension
        width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(False, True)
        activations_unskewed_direction_three_flipped = self.compute_multi_dimensional_lstm_one_direction(
            self.mdlstm_direction_three_parameters, width_flipping.flip(x))
        # Flip back the activations to get the retrieve the original orientation
        activations_unskewed_direction_three = width_flipping.flip(activations_unskewed_direction_three_flipped)

        # Flipping 2nd and 3th dimension combined
        height_and_width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, True)
        activations_unskewed_direction_four_flipped = self.compute_multi_dimensional_lstm_one_direction(
            self.mdlstm_direction_four_parameters, height_and_width_flipping.flip(x))
        # Flip back the activations to get the retrieve the original orientation
        activations_unskewed_direction_four = height_and_width_flipping.flip(activations_unskewed_direction_four_flipped)

        activations_combined = torch.cat((activations_unskewed_direction_one, activations_unskewed_direction_two,
                                          activations_unskewed_direction_three, activations_unskewed_direction_four), 1)

        result = activations_combined
        return result

    @staticmethod
    def compute_weighted_input_input_gate(column_number, input_gate_input_matrix, mdlstm_parameters):
        input_gate_input_column = input_gate_input_matrix[:, :, :, column_number]
        input_gate_hidden_state_column = mdlstm_parameters.get_input_gate_hidden_state_column()
        input_gate_memory_state_column = mdlstm_parameters.get_input_gate_memory_state_column()
        input_gate_weighted_states_plus_weighted_input = input_gate_input_column + \
            input_gate_hidden_state_column + input_gate_memory_state_column
        return input_gate_weighted_states_plus_weighted_input

    def compute_weighted_input_output_gate(self, mdlstm_parameters,
                                           previous_memory_state_column,
                                           column_number, output_gate_input_matrix):

        if self.use_dropout:
            output_gate_memory_state_column = \
                F.dropout(StateUpdateBlock.compute_weighted_state_input_state_one(
                    mdlstm_parameters.output_gate_memory_state_convolution,
                    previous_memory_state_column), p=0.2, training=self.training)
        else:
            output_gate_memory_state_column = StateUpdateBlock. \
                compute_weighted_state_input_state_one(mdlstm_parameters.output_gate_memory_state_convolution,
                                                       previous_memory_state_column)

        if self.clamp_gradients:
            output_gate_memory_state_column = InsideModelGradientClamping.register_gradient_clamping(output_gate_memory_state_column)

        return self.compute_weighted_input_forget_gate(
                mdlstm_parameters.get_output_gate_hidden_state_column(),
                output_gate_memory_state_column,
                column_number, output_gate_input_matrix)

    @staticmethod
    def compute_weighted_input_forget_gate(forget_gate_hidden_state_column,
                                           forget_gate_memory_state_column,
                                           column_number, forget_gate_input_matrix):

        forget_gate_input_column = forget_gate_input_matrix[:, :, :, column_number]
        forget_gate_weighted_states_plus_weighted_input = forget_gate_input_column + forget_gate_hidden_state_column + \
            forget_gate_memory_state_column

        # print("forget_gate_memory_state_column: " + str(forget_gate_memory_state_column))
        # print("forget_gate_hidden_state_column: " + str(forget_gate_hidden_state_column))
        # print("forget_gate_input_column: " + str(forget_gate_input_column))

        # print("forget_gate_weighted_states_plus_weighted_input: " + str(forget_gate_weighted_states_plus_weighted_input))

        return forget_gate_weighted_states_plus_weighted_input

    def forward_one_directional_multi_dimensional_lstm(self, x):
        activations_unskewed = self.compute_multi_dimensional_lstm_one_direction(self.mdlstm_direction_one_parameters,
                                                                                 x)
        # print("activations_unskewed.size(): " + str(activations_unskewed.size()))

        return activations_unskewed

    # This method computes the forward_one_directional_multi_dimensional_lstm but
    # adds one additional skewing and unskewing step. This is for testing that the
    # used methods for skewing and unskewing of the input do not mess up the gradient.
    def forward_one_directional_multi_dimensional_lstm_with_additional_skewing_unskewing_step(self, x):
        activations_unskewed = self.forward_one_directional_multi_dimensional_lstm(x)

        # Additional re-skewing step
        # print("activations_unskewed.size(): " + str(activations_unskewed.size()))
        activations_re_skewed = ImageInputTransformer.create_skewed_images_variable_four_dim(activations_unskewed)

        # print("activations_re_skewed.size(): " + str(activations_re_skewed.size()))
        original_image_columns = x.size(3)
        skewed_image_rows = x.size(2)

        # Additional re-un-skewing step
        activations_re_skewed_re_unskewed = ImageInputTransformer.\
            extract_unskewed_activations_from_activation_tensor(activations_re_skewed,
                                                                original_image_columns,
                                                                skewed_image_rows)
        # print("activations_re_skewed_re_unskewed.size(): " + str(activations_re_skewed_re_unskewed.size()))

        return activations_re_skewed_re_unskewed
        # return activations_unskewed

    # Needs to be implemented in the subclasses
    def _compute_multi_dimensional_function_one_direction(self, function_input):
        return self.compute_multi_dimensional_lstm_one_direction(self.mdlstm_direction_one_parameters, function_input)

    # Input tensor x is a batch of image tensors
    def forward(self, x):
        if self.compute_multi_directional_flag:
            # With distinct parameters for every direction
            return self.forward_multi_directional_multi_dimensional_lstm(x)
            # With same paramters for every direction
            #return self.forward_multi_directional_multi_dimensional_function_fast(x)
        else:
            return self.forward_one_directional_multi_dimensional_lstm(x)
            #return self.\
            #    forward_one_directional_multi_dimensional_lstm_with_additional_skewing_unskewing_step(x)
