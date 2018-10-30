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
from modules.multi_dimensional_lstm_parameters import MultiDirectionalMultiDimensionalLSTMParametersCreatorFullyParallel
from modules.multi_dimensional_lstm_parameters import \
    MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution
from modules.multi_dimensional_lstm_parameters import \
    MultiDirectionalMultiDimensionalLSTMParametersCreatorParallelWithSeparateInputConvolution
from modules.multi_dimensional_lstm_parameters import \
    MultiDirectionalMultiDimensionalLSTMParametersFullyParallel
from modules.multi_dimensional_lstm_parameters import \
    MultiDirectionalMultiDimensionalLeakyLPCellParametersCreatorFullyParallel
from util.image_input_transformer import ImageInputTransformer
from modules.inside_model_gradient_clipping import InsideModelGradientClamping
from util.tensor_utils import TensorUtils
from modules.mdlstm_examples_packing import MDLSTMExamplesPacking


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

    def __init__(self, layer_index: int, input_channels: int, hidden_states_size: int, compute_multi_directional: bool,
                 clamp_gradients: bool,
                 use_dropout: bool, training: bool,
                 mdlstm_parameters,
                 use_example_packing: bool,
                 nonlinearity="tanh"):
        super(MultiDimensionalLSTM, self).__init__(layer_index, input_channels, hidden_states_size,
                                                   compute_multi_directional,
                                                   nonlinearity)

        self.clamp_gradients = clamp_gradients
        if self.clamp_gradients:
            print("MultiDimensionalLSTM - clamp_gradients=" + str(self.clamp_gradients))
        else:
            print("WARNING: MultiDimensionalLSTM - clamp_gradients=" + str(self.clamp_gradients))

        self.use_dropout = use_dropout
        self.training = training
        self.use_example_packing = use_example_packing
        self.mdlstm_parameters =  mdlstm_parameters

        # # For multi-directional rnn
        # if self.compute_multi_directional_flag:
        #     self.mdlstm_direction_two_parameters = \
        #         multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
        #             self.hidden_states_size, self.input_channels, self.clamp_gradients, use_dropout)
        #     # Set initial bias for the forget gates to one, since it is known to give better results
        #     self.mdlstm_direction_two_parameters.set_bias_forget_gates_to_one()
        #
        #     self.mdlstm_direction_three_parameters = \
        #         multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
        #             self.hidden_states_size, self.input_channels, self.clamp_gradients, use_dropout)
        #     # Set initial bias for the forget gates to one, since it is known to give better results
        #     self.mdlstm_direction_three_parameters.set_bias_forget_gates_to_one()
        #
        #     self.mdlstm_direction_four_parameters = \
        #         multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
        #             self.hidden_states_size, self.input_channels, self.clamp_gradients, use_dropout)
        #     # Set initial bias for the forget gates to one, since it is known to give better results
        #     self.mdlstm_direction_four_parameters.set_bias_forget_gates_to_one()

        # See: https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
        # self.register_backward_hook(printgradnorm)

    @staticmethod
    def create_mdlstm_paramters(multi_dimensional_lstm_parameter_creator,
                                compute_multi_directional: bool,
                                hidden_states_size: int, input_channels: int,
                                use_dropout: bool, clamp_gradients: bool):
        if compute_multi_directional:
            mdlstm_parameters = \
                multi_dimensional_lstm_parameter_creator.create_multi_directional_multi_dimensional_lstm_parameters(
                    hidden_states_size, input_channels,
                    use_dropout, clamp_gradients, 4)
        else:

            mdlstm_parameters = \
                multi_dimensional_lstm_parameter_creator.create_multi_dimensional_lstm_parameters_one_direction(
                    hidden_states_size, input_channels, clamp_gradients, use_dropout)

        # Set initial bias for the forget gates to one, since it is known to give better results
        mdlstm_parameters.set_bias_forget_gates_to_one()

        return mdlstm_parameters

    @staticmethod
    def create_multi_dimensional_lstm(layer_index: int, input_channels: int, hidden_states_size: int,
                                      compute_multi_directional: bool,
                                      clamp_gradients: bool,
                                      use_dropout: bool,
                                      use_example_packing: bool,
                                      nonlinearity="tanh"):

        mdlstm_parameters = MultiDimensionalLSTM.create_mdlstm_paramters(
            MultiDimensionalLSTMParametersCreatorSlow(),
            compute_multi_directional, hidden_states_size, input_channels,
            use_dropout, clamp_gradients)

        return MultiDimensionalLSTM(layer_index, input_channels, hidden_states_size, compute_multi_directional,
                                    clamp_gradients, use_dropout,
                                    True,
                                    mdlstm_parameters,
                                    use_example_packing,
                                    nonlinearity)

    @staticmethod
    def create_multi_dimensional_lstm_fast(layer_index: int, input_channels: int, hidden_states_size: int,
                                           compute_multi_directional: bool,
                                           clamp_gradients: bool,
                                           use_dropout: bool,
                                           use_example_packing: bool,
                                           nonlinearity="tanh"):

        mdlstm_parameters = MultiDimensionalLSTM.create_mdlstm_paramters(
            MultiDimensionalLSTMParametersCreatorFast(),
            compute_multi_directional, hidden_states_size, input_channels,
            use_dropout, clamp_gradients)

        return MultiDimensionalLSTM(layer_index, input_channels, hidden_states_size, compute_multi_directional,
                                    clamp_gradients, use_dropout,
                                    True,
                                    mdlstm_parameters,
                                    use_example_packing,
                                    nonlinearity)

    @staticmethod
    def create_multi_dimensional_lstm_parallel_with_separate_input_convolution(
            layer_index: int, input_channels: int, hidden_states_size: int,
            compute_multi_directional: bool,
            clamp_gradients: bool,
            use_dropout: bool,
            use_example_packing: bool,
            nonlinearity="tanh"):

        if compute_multi_directional:
            mult_dimensional_lstm_parameters_creater = \
                MultiDirectionalMultiDimensionalLSTMParametersCreatorParallelWithSeparateInputConvolution()
        else:
            mult_dimensional_lstm_parameters_creater = MultiDimensionalLSTMParametersCreatorFast()

        mdlstm_parameters = MultiDimensionalLSTM.create_mdlstm_paramters(
            mult_dimensional_lstm_parameters_creater,
            compute_multi_directional, hidden_states_size, input_channels,
            use_dropout, clamp_gradients)

        return MultiDimensionalLSTM(layer_index, input_channels, hidden_states_size, compute_multi_directional,
                                    clamp_gradients, use_dropout,
                                    True,
                                    mdlstm_parameters,
                                    use_example_packing,
                                    nonlinearity)

    @staticmethod
    def create_multi_dimensional_lstm_fully_parallel(
            layer_index: int, input_channels: int, hidden_states_size: int,
            compute_multi_directional: bool,
            clamp_gradients: bool,
            use_dropout: bool,
            use_example_packing: bool,
            use_leaky_lp_cells: bool,
            nonlinearity="tanh"):

        if compute_multi_directional:
            if use_leaky_lp_cells:
                mult_dimensional_lstm_parameters_creater = \
                    MultiDirectionalMultiDimensionalLeakyLPCellParametersCreatorFullyParallel()
            else:
                mult_dimensional_lstm_parameters_creater = \
                    MultiDirectionalMultiDimensionalLSTMParametersCreatorFullyParallel()
        else:
            raise RuntimeError("Not implemented")

        mdlstm_parameters = MultiDimensionalLSTM.create_mdlstm_paramters(
            mult_dimensional_lstm_parameters_creater,
            compute_multi_directional, hidden_states_size, input_channels,
            use_dropout, clamp_gradients)

        return MultiDimensionalLSTM(layer_index, input_channels, hidden_states_size, compute_multi_directional,
                                    clamp_gradients, use_dropout,
                                    True,
                                    mdlstm_parameters,
                                    use_example_packing,
                                    nonlinearity)

    def create_one_directional_mdlstms_from_multi_directional_mdlstm(self):
        if not self.compute_multi_directional():
            raise RuntimeError("Error: only allowed for multi-directional MDLSTM")
        if not (isinstance(self.mdlstm_parameters,
                          MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution) \
                or isinstance(self.mdlstm_parameters,
                          MultiDirectionalMultiDimensionalLSTMParametersFullyParallel)):
            raise RuntimeError("Error: method only implemented for multi-directional MDSLTM with "
                               "parameters of type MultiDirectionalMultiDimensionalLSTMParametersFullyParallel"
                               "or of type  "
                               "MultiDirectionalMultiDimensionalLSTMParametersParallelWithSeparateInputConvolution")

        result = list([])

        one_directional_mdlstms_parameters = self.mdlstm_parameters.\
            create_one_directional_mdlstm_parameters_each_direction_using_current_weights()
        for mdlstm_parameters in one_directional_mdlstms_parameters:
            result.append(MultiDimensionalLSTM(self.layer_index, self.input_channels, self.hidden_states_size,
                          False, self.clamp_gradients, self.use_dropout, self.training,
                          mdlstm_parameters, self.use_example_packing, self.nonlinearity))
        return result

    def set_training(self, training):
        self.mdlstm_parameters.set_training(training)

        # if self.compute_multi_directional_flag:
        #     self.mdlstm_direction_two_parameters.set_training(training)
        #     self.mdlstm_direction_three_parameters.set_training(training)
        #     self.mdlstm_direction_four_parameters.set_training(training)

        self.training = training

    def prepare_skewed_images_and_mask(self, examples):
        if self.use_example_packing:
            mdlstm_examples_packing = \
                MDLSTMExamplesPacking.created_mdlstm_examples_packing(examples, 1)
            if self.compute_multi_directional():
                # time_start_packing = util.timing.date_time_start()

                skewed_images_variable, mask = mdlstm_examples_packing. \
                    create_vertically_and_horizontally_packed_examples_four_directions_plus_mask(examples)
                if skewed_images_variable.size(1) != 4 * self.input_channels:
                    raise RuntimeError("Error: expected the 4 images for four directions to be stacked on "
                                       "the second (channel) dimension")
                number_of_images = 4

                # print("multi_dimensional_lstm - Time used for examples packing: "
                #      + str(util.timing.milliseconds_since(time_start_packing)))

            else:
                skewed_images_variable, mask = mdlstm_examples_packing.\
                    create_vertically_and_horizontally_packed_examples_and_mask_one_direction(examples)
                number_of_images = 1
        else:
            # Create a binary mask that tells which of the cell positions are valid and which are not
            skewed_images_variable = ImageInputTransformer.create_skewed_images_variable_four_dim(examples)
            mask = ImageInputTransformer.create_skewed_images_mask_two_dim(examples)
            number_of_images = examples.size(0)
            mdlstm_examples_packing = None
        return skewed_images_variable, mask, number_of_images, mdlstm_examples_packing

    def prepare_initial_states(self, image_height: int, number_of_images: int, device):
        if self.compute_multi_directional():
            # print("image height: " + str(image_height))
            previous_hidden_state_column = torch.zeros(1,
                                                       self.hidden_states_size * 4,
                                                       image_height)

            # and previous_hidden_state: why the latter has dimension equal to
            # batch size but for the former it doesn't seem to matter
            previous_memory_state_column = torch.zeros(1,
                                                       self.hidden_states_size * 4,
                                                       image_height)
        else:
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
        return previous_hidden_state_column, previous_memory_state_column

    def compute_multi_dimensional_lstm(self, mdlstm_parameters, examples):

        skewed_images_variable, mask, number_of_images, mdlstm_examples_packing = self.prepare_skewed_images_and_mask(examples)

        # print("skewed_images_variable: " + str(skewed_images_variable))

        # Add a column of padding zeros to mask, so that mask[:, column_index]
        # will return the padding for the previous column
        p2d = (1, 0, 0, 0)
        mask = torch.nn.functional.pad(mask, p2d, "constant", 0)

        if MultiDimensionalRNNBase.use_cuda():
            # https://discuss.pytorch.org/t/which-device-is-model-tensor-stored-on/4908/7
            device = skewed_images_variable.get_device()

        # print("compute_multi_dimensional_lstm_one_direction - x.size(): " + str(x.size()))
        # print("compute_multi_dimensional_lstm_one_direction - self.hidden_states_size: " + str(self.hidden_states_size))

        # print("list(x.size()): " + str(list(x.size())))
        image_height = skewed_images_variable.size(2)

        previous_hidden_state_column, previous_memory_state_column = self.prepare_initial_states(
            image_height, number_of_images, device)

        skewed_image_columns = skewed_images_variable.size(3)

        # print("mdlstm_parameters.input_input_convolution: " + str(mdlstm_parameters.input_input_convolution))
        # print("skewed_images_variable.get_device(): " + str(skewed_images_variable.get_device()))
        # print("mdlstm_parameters.input_input_convolution.bias: "
        # + str(mdlstm_parameters.input_input_convolution.bias))

        # Prepare input convolutions if applicable
        mdlstm_parameters.prepare_input_convolutions(skewed_images_variable)

        # if self.clamp_gradients:
        #     # print("MultiDimensionalLSTM.compute_multi_dimensional_lstm_one_direction - register gradient clamping...")
        #     input_input_matrix = InsideModelGradientClamping.register_gradient_clamping(input_input_matrix)
        #     input_gate_input_matrix = InsideModelGradientClamping.register_gradient_clamping(input_gate_input_matrix)
        #     forget_gate_one_input_matrix = InsideModelGradientClamping.register_gradient_clamping(forget_gate_one_input_matrix)
        #     forget_gate_two_input_matrix = InsideModelGradientClamping.register_gradient_clamping(forget_gate_two_input_matrix)
        #     output_gate_input_matrix = InsideModelGradientClamping.register_gradient_clamping(output_gate_input_matrix)

        activations = list([])

        # print("skewed image columns: " + str(skewed_image_columns))

        # print("starting MDLSTM column computation...")

        for column_index in range(0, skewed_image_columns):
            # print("column_index: " + str(column_index))
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

            # Apply a binary mask to zero out entries in the activation_column
            # and new_memory_state that are not corresponding to valid states,
            # but that are an artifact of the computation by convolution using
            # the image skewing trick

            valid_entries_selection_mask_previous_column = mask[:, column_index]
            valid_entries_selection_mask = mask[:, column_index + 1]
            # print("valid_entries_selection_mask: " +
            # str(valid_entries_selection_mask))

            mdlstm_parameters.prepare_computation_next_column_functions(previous_hidden_state_column,
                                                                        previous_memory_state_column,
                                                                        valid_entries_selection_mask_previous_column)

            # Compute convolution on previous state column vector padded with zeros
            # Compute convolution on previous state column vector padded with zeros
            input_hidden_state_column = mdlstm_parameters.get_input_hidden_state_column()

            # print("input_hidden_state_column.size(): " + str(input_hidden_state_column.size()))
            # print("input_hidden_state_column: " + str(input_hidden_state_column))

            input_state_plus_input = MultiDimensionalRNNBase.\
                compute_states_plus_input(mdlstm_parameters.get_input_input_column(column_index),
                                          input_hidden_state_column)

            # Compute the sum of weighted inputs of the input gate
            input_gate_input_column = mdlstm_parameters.get_input_gate_input_column(column_index)
            input_gate_weighted_states_plus_input = MultiDimensionalLSTM.\
                compute_weighted_input_input_gate(input_gate_input_column,
                                                  mdlstm_parameters)

            # Clamp before activation functions
            if self.clamp_gradients:
                input_gate_weighted_states_plus_input = \
                     InsideModelGradientClamping.register_gradient_clamping_default_clamping_bound(
                         input_gate_weighted_states_plus_input, "mdlstm - input_gate_weighted_states_plus_input")
                input_state_plus_input = InsideModelGradientClamping.\
                    register_gradient_clamping_default_clamping_bound(input_state_plus_input,
                                                                      "mdlstm - input_states_plus_input")

            # Compute the input activation
            input_activation_column = F.tanh(input_state_plus_input)
            #input_activation_column = F.relu(input_state_plus_input) # Relu can be used as an alternative to tanh
            # Compute the input gate activation
            input_gate_activation_column = F.sigmoid(input_gate_weighted_states_plus_input)

            input_and_input_gate_combined = torch.mul(input_activation_column, input_gate_activation_column)

            # print("input_and_input_gate_combined.size(): " + str(input_and_input_gate_combined.size()))

            if self.clamp_gradients:
                input_and_input_gate_combined = \
                    InsideModelGradientClamping.\
                        register_gradient_clamping(input_and_input_gate_combined, 10, False,
                                                   "mdlstm - input_and_input_gate_combined")


            # print("input and input gate combined: " + str(input_and_input_gate_combined))

            memory_states_column_forget_gate_one = previous_memory_state_column

            forget_gate_one_input_column = mdlstm_parameters.get_forget_gate_one_input_column(column_index)
            forget_gate_one_weighted_states_plus_input = self.compute_weighted_input_forget_gate(
                mdlstm_parameters.get_forget_gate_one_hidden_state_column(),
                mdlstm_parameters.get_forget_gate_one_memory_state_column(),
                forget_gate_one_input_column)

            if self.clamp_gradients:
                forget_gate_one_weighted_states_plus_input = \
                    InsideModelGradientClamping.register_gradient_clamping_default_clamping_bound(
                        forget_gate_one_weighted_states_plus_input,
                        "mdlstm - forget_gate_one_weighted_states_plus_input")


            # print(">>> forget_gate_one_weighted_states_plus_input: " + str(forget_gate_one_weighted_states_plus_input))

            # Compute the forget gate one activation
            forget_gate_one_activation_column = F.sigmoid(forget_gate_one_weighted_states_plus_input)
            # print("forget gate one activation column: " + str(forget_gate_one_activation_column))

            # Compute the activation for forget gate one
            forget_gate_one_activation_multiplied_with_previous_memory_state = \
                torch.mul(forget_gate_one_activation_column,
                          memory_states_column_forget_gate_one)

            if self.clamp_gradients:
                forget_gate_one_activation_multiplied_with_previous_memory_state = \
                    InsideModelGradientClamping.register_gradient_clamping(
                        forget_gate_one_activation_multiplied_with_previous_memory_state, 10, False,
                        "mdlstm - forget_gate_one_activation_multiplied_with_previous_memory_state")

            memory_states_column_forget_gate_two = StateUpdateBlock.\
                get_shifted_column_fast(previous_memory_state_column,
                                        self.clamp_gradients)

            forget_gate_two_input_column = mdlstm_parameters.get_forget_gate_two_input_column(column_index)
            forget_gate_two_weighted_states_plus_input = self.compute_weighted_input_forget_gate(
                mdlstm_parameters.get_forget_gate_two_hidden_state_column(),
                mdlstm_parameters.get_forget_gate_two_memory_state_column(),
                forget_gate_two_input_column)

            if self.clamp_gradients:
                forget_gate_two_weighted_states_plus_input = \
                    InsideModelGradientClamping.register_gradient_clamping_default_clamping_bound(
                        forget_gate_two_weighted_states_plus_input,
                        "mdlstm - forget_gate_two_weighted_states_plus_input")

            # Compute the forget gate two activation
            forget_gate_two_activation_column = F.sigmoid(forget_gate_two_weighted_states_plus_input)

            # TensorUtils.print_max(forget_gate_one_activation_column, "forget_gate_one_activation_column")
            # TensorUtils.print_max(forget_gate_two_activation_column, "forget_gate_two_activation_column")


            # forget_gate_weighted_states_combined =  forget_gate_one_weighted_stated_plus_input + forget_gate_two_weighted_stated_plus_input
            # forget_gates_combined_activation_column = F.sigmoid(forget_gate_weighted_states_combined)
            # forget_gates_combined_activation_multiplied_with_previous_memory_state = torch.mul(
            #    forget_gates_combined_activation_column, previous_memory_state_column)

            # Compute the activation for forget gate two
            forget_gate_two_activation_multiplied_with_previous_memory_state = torch.mul(
                forget_gate_two_activation_column, memory_states_column_forget_gate_two)

            if self.clamp_gradients:
                forget_gate_two_activation_multiplied_with_previous_memory_state = \
                    InsideModelGradientClamping.register_gradient_clamping(
                        forget_gate_two_activation_multiplied_with_previous_memory_state, 10, False,
                        "mdlstm - forget_gate_two_activation_multiplied_with_previous_memory_state")

            # print("input_and_input_gate_combined: " + str(input_and_input_gate_combined))

            # print("forget_gate_one_activation_column: " + str(forget_gate_two_activation_column))
            # print("memory_states_column_forget_gate_one: " + str(memory_states_column_forget_gate_one))
            # print("forget_gate_two_activation_column: " + str(forget_gate_two_activation_column))
            #print("memory_states_column_forget_gate_two: " + str(memory_states_column_forget_gate_two))
            #print("forget_gate_one_activation_multiplied_with_previous_memory_state: "+
            #      str(forget_gate_one_activation_multiplied_with_previous_memory_state))
            #print("forget_gate_two_activation_multiplied_with_previous_memory_state: " +
            #      str(forget_gate_two_activation_multiplied_with_previous_memory_state))


            # Multiplying the values of forget_gate_two_activation_multiplied_with_previous_memory_state
            # and forget_gate_one_activation_multiplied_with_previous_memory_state with factor 0.5
            # is a hack to avoid that the new_memory_state cannot grow unbounded
            # Since the gate activations can be up to 1, summing two functions of previous memory state
            # without this normalization leads to a memory state that can keep growing
            new_memory_state = input_and_input_gate_combined + \
                0.5 * forget_gate_two_activation_multiplied_with_previous_memory_state + \
                0.5 * forget_gate_one_activation_multiplied_with_previous_memory_state # + \
                # forget_gates_combined_activation_multiplied_with_previous_memory_state \

            # # As an alternative, with the bias of the forget gates initialized to 0, but does not seem to work
            # # (still produces nans)
            # new_memory_state = input_and_input_gate_combined + \
            #                    forget_gate_two_activation_multiplied_with_previous_memory_state + \
            #                    forget_gate_one_activation_multiplied_with_previous_memory_state  # + \
            # forget_gates_combined_activation_multiplied_with_previous_memory_state \

            # print("new_memory_state.requires_grad: " + str(new_memory_state.requires_grad))

            if self.clamp_gradients:
                new_memory_state = \
                    InsideModelGradientClamping.register_gradient_clamping_default_clamping_bound(
                        new_memory_state,
                        "mdlstm - new_memory_state")

            #new_memory_state = input_and_input_gate_combined + \
            #    forget_gate_two_activation_multiplied_with_previous_memory_state

            # print("new memory state: " + str(new_memory_state))

            # This additional tanh activation function taken from the NVIDIA diagram
            # was not in the deep learning book diagram, and does not seem to help
            # really ?
            new_memory_state_activation_column = F.tanh(new_memory_state)

            # This appears to be the first gradient component that gets nan input on the backward pass
            variable_identification_string = "layer: " + str(self.layer_index) + " column_index: " \
                                             + str(column_index) + " (max index = " + str(skewed_image_columns - 1) \
                                             + ") - mdlstm - new_memory_state_activation_column"
            if self.clamp_gradients:
                new_memory_state_activation_column = InsideModelGradientClamping.\
                    register_gradient_clamping(
                        new_memory_state_activation_column, 10, True, variable_identification_string
                    )

            # This grows too much in the forward pass unless new_memory_state computation
            # multiplies contributions of memory states each by factor 0.5
            # TensorUtils.print_max(new_memory_state, "new_memory_state")


            # Compute the sum of weighted inputs of the ouput gate
            output_gate_input_column = mdlstm_parameters.get_output_gate_input_column(column_index)
            output_gate_weighted_states_plus_input = self. \
                compute_weighted_input_output_gate(mdlstm_parameters, new_memory_state,
                                                   output_gate_input_column)

            if self.clamp_gradients:
                output_gate_weighted_states_plus_input = \
                    InsideModelGradientClamping.register_gradient_clamping_default_clamping_bound(
                        output_gate_weighted_states_plus_input,
                    "mdlstm - output_gate_weighted_states_plus_input")

            # This grows too much in the forward pass unless new_memory_state computation
            # multiplies contributions of memory states each by factor 0.5
            # e.g. without correction:
            # max element in output_gate_weighted_states_plus_input :tensor(1.4322e+10, device='cuda:0')
            # TensorUtils.print_max(output_gate_weighted_states_plus_input, "output_gate_weighted_states_plus_input")

            # output_gate_activation_column = F.sigmoid(output_gate_weighted_states_plus_input)
            output_gate_activation_column = torch.sigmoid(output_gate_weighted_states_plus_input)


            # print("output_gate_activation_column.requires_grad:" + str(output_gate_activation_column.requires_grad))

            # This appears to be the first gradient component that gets nan input on the backward pass
            # Could one of the gradients be None? https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887/5
            # https://github.com/pytorch/pytorch/issues/4132
            variable_identification_string = "layer: " + str(self.layer_index) + " column_index: " \
                                             + str(column_index) + " (max index = " + str(skewed_image_columns - 1) \
                                             + ") - mdlstm - output_gate_activation_column"
            if self.clamp_gradients:
                output_gate_activation_column = InsideModelGradientClamping.\
                    register_gradient_clamping_default_clamping_bound(
                        output_gate_activation_column, variable_identification_string
                    )



            # TensorUtils.print_max(output_gate_activation_column, "output_gate_activation_column")

            # print("input_column: " + str(input_column))
            #print("state_plus_input: " + str(state_plus_input))

            # This is according to the NVIDIA LSTM diagram
            # activation_column = torch.mul(new_memory_state_activation_column, output_gate_activation_column)

            # # This is following the deep learning book
            activation_column = torch.mul(new_memory_state, output_gate_activation_column)

            if self.clamp_gradients:
                InsideModelGradientClamping.register_gradient_clamping(activation_column, 10, True,
                                                                       "mdlstm - activation_column")

            #activation_column = self.get_activation_function()(input_state_plus_input)
            # activation_column = new_memory_state_activation_column
            # print("output gate activation column: " + str(output_gate_activation_column))
            #print("activation column: " + str(activation_column))

            # Apply the selection mask to the activation column and new_memory_state
            # This will set to zero the activation and memory states of masked
            # (non-valid) cells, effectively resetting them for the computation
            # in the next column cells that will use them as memory and hidden state
            # inputs
            activation_column = TensorUtils.apply_binary_mask(activation_column, valid_entries_selection_mask)
            new_memory_state = TensorUtils.apply_binary_mask(new_memory_state, valid_entries_selection_mask)

            previous_hidden_state_column = activation_column
            previous_memory_state_column = new_memory_state

            # Does not seem to help either: https://github.com/pytorch/pytorch/issues/4649
            # previous_hidden_state_column.retain_grad()
            # previous_memory_state_column.retain_grad()

            activations.append(activation_column)

            # In the loop the value of grad_fn becomes set, as a backwards path for
            # back-propagation is collected
            # print("in loop: previous_memory_state_column.grad_fn: " + str(previous_memory_state_column.grad_fn))
            # print("in loop: previous_hidden_state_column.grad_fn: " + str(previous_hidden_state_column.grad_fn))

        if self.use_example_packing:
            # time_start_unpacking = util.timing.date_time_start()
            activations_unskewed = mdlstm_examples_packing.\
                extract_unskewed_examples_activations_from_activation_columns(activations)
            # print("len(activations_unskewed: " + str(len(activations_unskewed)))
            # for tensor in activations_unskewed:
            #     print("MDLSTM with packing output activations tensor size: " + str(tensor.size()))
            # print("multi_dimensional_lstm - Time used for examples unpacking: "
            #       + str(util.timing.milliseconds_since(time_start_unpacking)))
        else:
            # print(">>> examples.size(): " + str(examples.size()))
            original_image_columns = examples.size(3)
            skewed_image_rows = skewed_images_variable.size(2)

            activations_unskewed = ImageInputTransformer.\
                extract_unskewed_activations_from_activation_columns(activations, original_image_columns)

        # print("activations_unskewed: " + str(activations_unskewed))
        return activations_unskewed

        # This function is slow because all four function calls for 4 directions are
        # executed sequentially. It isn't entirely clear how to optimize this.
        # See the discussion at:
        # https://discuss.pytorch.org/t/is-there-a-way-to-parallelize-independent-sequential-steps/3360

    def compute_leaky_lp_cell(self, mdlstm_parameters, examples):

        skewed_images_variable, mask, number_of_images, mdlstm_examples_packing = self.prepare_skewed_images_and_mask(
            examples)

        # Add a column of padding zeros to mask, so that mask[:, column_index]
        # will return the padding for the previous column
        p2d = (1, 0, 0, 0)
        mask = torch.nn.functional.pad(mask, p2d, "constant", 0)

        if MultiDimensionalRNNBase.use_cuda():
            # https://discuss.pytorch.org/t/which-device-is-model-tensor-stored-on/4908/7
            device = skewed_images_variable.get_device()

        image_height = skewed_images_variable.size(2)

        previous_hidden_state_column, previous_memory_state_column = self.prepare_initial_states(
            image_height, number_of_images, device)

        skewed_image_columns = skewed_images_variable.size(3)

        # Prepare input convolutions if applicable
        mdlstm_parameters.prepare_input_convolutions(skewed_images_variable)

        activations = list([])

        # print("skewed image columns: " + str(skewed_image_columns))
        # print("starting Leaky LP cell column computation...")

        for column_index in range(0, skewed_image_columns):
            # Apply a binary mask to zero out entries in the activation_column
            # and new_memory_state that are not corresponding to valid states,
            # but that are an artifact of the computation by convolution using
            # the image skewing trick

            valid_entries_selection_mask_previous_column = mask[:, column_index]
            valid_entries_selection_mask = mask[:, column_index + 1]
            # print("valid_entries_selection_mask: " +
            # str(valid_entries_selection_mask))

            mdlstm_parameters.prepare_computation_next_column_functions(previous_hidden_state_column,
                                                                        previous_memory_state_column,
                                                                        valid_entries_selection_mask_previous_column)

            # Compute convolution on previous state column vector padded with zeros
            input_hidden_state_column = mdlstm_parameters.get_input_hidden_state_column()

            input_state_plus_input = MultiDimensionalRNNBase. \
                compute_states_plus_input(mdlstm_parameters.get_input_input_column(column_index),
                                          input_hidden_state_column)

            # Compute the sum of weighted inputs of the input gate
            input_gate_input_column = mdlstm_parameters.get_input_gate_input_column(column_index)
            input_and_states_lambda_gate_weighted_states_plus_input = MultiDimensionalLSTM. \
                compute_weighted_input_input_gate(input_gate_input_column,
                                                  mdlstm_parameters)

            # Compute the input activation
            input_activation_column = F.tanh(input_state_plus_input)
            # input_activation_column = F.relu(input_state_plus_input) # Relu can be used as an alternative to tanh
            # Compute the input gate activation
            input_and_states_lambda_gate_activation_column_input = F.sigmoid(
                input_and_states_lambda_gate_weighted_states_plus_input)

            input_and_states_lambda_gate_activation_column_states = \
                torch.ones_like(input_and_states_lambda_gate_activation_column_input) - \
                input_and_states_lambda_gate_activation_column_input

            memory_states_column_forget_gate_one = previous_memory_state_column

            states_lambda_gate_input_column = mdlstm_parameters.get_forget_gate_one_input_column(column_index)
            states_lambda_gate_weighted_states_plus_input = MultiDimensionalLSTM.compute_weighted_input_lambda_gate(
                mdlstm_parameters.get_forget_gate_one_hidden_state_column(),
                mdlstm_parameters.get_forget_gate_one_memory_state_column(),
                mdlstm_parameters.get_forget_gate_two_memory_state_column(),
                states_lambda_gate_input_column)

            states_lambda_gate_activation_column_state_one = F.sigmoid(states_lambda_gate_weighted_states_plus_input)
            # print("states lambda gate activation column_S1: " +
            #       str(states_lambda_gate_activation_column_state_one))

            states_lambda_gate_activation_column_state_two = \
                torch.ones_like(states_lambda_gate_activation_column_state_one) - \
                states_lambda_gate_activation_column_state_one
            # print("states lambda gate activation column_S2: " +
            #       str(states_lambda_gate_activation_column_state_two))

            memory_states_column_forget_gate_two = StateUpdateBlock. \
                get_shifted_column_fast(previous_memory_state_column,
                                        self.clamp_gradients)

            # Compute the re-weighted (i.e.) mixed state produced
            # by the states lambda gate
            states_lambda_gate_reweighted_memory_states = \
                torch.mul(states_lambda_gate_activation_column_state_one,
                          memory_states_column_forget_gate_one) +\
                torch.mul(states_lambda_gate_activation_column_state_two,
                          memory_states_column_forget_gate_two)

            new_memory_state = \
                torch.mul(input_activation_column, input_and_states_lambda_gate_activation_column_input) +\
                torch.mul(states_lambda_gate_reweighted_memory_states,
                          input_and_states_lambda_gate_activation_column_states)

            output_gates_memory_state_column = \
                mdlstm_parameters.compute_output_gate_memory_state_weighted_input(previous_memory_state_column)
            # print(">>>> output_gates_memory_state_column: " + str(output_gates_memory_state_column))
            # print(">>>> output_gates_memory_state_column.size(): " + str(output_gates_memory_state_column.size()))
            output_gates_memory_state_columns = torch.chunk(output_gates_memory_state_column, 2, 1)
            output_gate_one_memory_state_column = output_gates_memory_state_columns[0]
            output_gate_two_memory_state_column = output_gates_memory_state_columns[1]

            output_gate_one_input_column = mdlstm_parameters.\
                get_forget_gate_two_input_column(column_index)
            output_gate_one_weighted_states_plus_input = \
                MultiDimensionalLSTM.compute_weighted_input_forget_gate(
                    mdlstm_parameters.get_forget_gate_two_hidden_state_column(),
                    output_gate_one_memory_state_column,
                    output_gate_one_input_column)

            output_gate_two_input_column = mdlstm_parameters.get_output_gate_input_column(column_index)
            output_gate_two_weighted_states_plus_input = \
                MultiDimensionalLSTM.compute_weighted_input_forget_gate(
                    mdlstm_parameters.get_output_gate_hidden_state_column(),
                    output_gate_two_memory_state_column,
                    output_gate_two_input_column)

            # Compute the output gate one activation
            output_gate_one_activation_column = F.sigmoid(output_gate_one_weighted_states_plus_input)
            # Compute the output gate two activation
            output_gate_two_activation_column = F.sigmoid(output_gate_two_weighted_states_plus_input)

            output_gate_one_output = torch.mul(states_lambda_gate_reweighted_memory_states,
                                               output_gate_one_activation_column)

            output_gate_two_output = torch.mul(new_memory_state,
                                               output_gate_two_activation_column)

            output_gates_combined_output = output_gate_one_output + output_gate_two_output

            # With final tanh as in the NVIDIA LSTM diagram
            activation_column = F.tanh(output_gates_combined_output)

            # # This is following the deep learning book
            # activation_column = output_gates_combined_output

            # Apply the selection mask to the activation column and new_memory_state
            # This will set to zero the activation and memory states of masked
            # (non-valid) cells, effectively resetting them for the computation
            # in the next column cells that will use them as memory and hidden state
            # inputs
            activation_column = TensorUtils.apply_binary_mask(activation_column, valid_entries_selection_mask)
            new_memory_state = TensorUtils.apply_binary_mask(new_memory_state, valid_entries_selection_mask)

            previous_hidden_state_column = activation_column
            previous_memory_state_column = new_memory_state

            # Does not seem to help either: https://github.com/pytorch/pytorch/issues/4649
            # previous_hidden_state_column.retain_grad()
            # previous_memory_state_column.retain_grad()

            activations.append(activation_column)

            # In the loop the value of grad_fn becomes set, as a backwards path for
            # back-propagation is collected
            # print("in loop: previous_memory_state_column.grad_fn: " + str(previous_memory_state_column.grad_fn))
            # print("in loop: previous_hidden_state_column.grad_fn: " + str(previous_hidden_state_column.grad_fn))

        if self.use_example_packing:
            # time_start_unpacking = util.timing.date_time_start()
            activations_unskewed = mdlstm_examples_packing. \
                extract_unskewed_examples_activations_from_activation_columns(activations)
            # print("len(activations_unskewed: " + str(len(activations_unskewed)))
            # for tensor in activations_unskewed:
            #     print("MDLSTM with packing output activations tensor size: " + str(tensor.size()))
            # print("multi_dimensional_lstm - Time used for examples unpacking: "
            #       + str(util.timing.milliseconds_since(time_start_unpacking)))
        else:
            # print(">>> x.size(): " + str(x.size()))
            original_image_columns = examples.size(3)
            skewed_image_rows = skewed_images_variable.size(2)

            activations_unskewed = ImageInputTransformer. \
                extract_unskewed_activations_from_activation_columns(activations, original_image_columns)

        # print("activations_unskewed: " + str(activations_unskewed))
        return activations_unskewed

    def forward_multi_directional_multi_dimensional_lstm(self, x):

        # time_start_network_forward = util.timing.date_time_start()

        # # print("list(x.size()): " + str(list(x.size())))
        #
        # # Original order
        # activations_unskewed_direction_one = self.\
        #     compute_multi_dimensional_lstm_one_direction(self.mdlstm_parameters, x)
        #
        #
        # # Flipping 2nd dimension
        # height_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, False)
        # activations_unskewed_direction_two_flipped = self.compute_multi_dimensional_lstm_one_direction(
        #     self.mdlstm_direction_two_parameters, height_flipping.flip(x))
        # # Flip back the activations to get the retrieve the original orientation
        # activations_unskewed_direction_two = height_flipping.flip(activations_unskewed_direction_two_flipped)
        #
        # # print("activations_one_dimensional_two: " + str(activations_one_dimensional_two))
        #
        # # Flipping 3th dimension
        # width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(False, True)
        # activations_unskewed_direction_three_flipped = self.compute_multi_dimensional_lstm_one_direction(
        #     self.mdlstm_direction_three_parameters, width_flipping.flip(x))
        # # Flip back the activations to get the retrieve the original orientation
        # activations_unskewed_direction_three = width_flipping.flip(activations_unskewed_direction_three_flipped)
        #
        # # Flipping 2nd and 3th dimension combined
        # height_and_width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, True)
        # activations_unskewed_direction_four_flipped = self.compute_multi_dimensional_lstm_one_direction(
        #     self.mdlstm_direction_four_parameters, height_and_width_flipping.flip(x))
        # # Flip back the activations to get the retrieve the original orientation
        # activations_unskewed_direction_four = height_and_width_flipping.flip(activations_unskewed_direction_four_flipped)
        #
        # activations_combined = torch.cat((activations_unskewed_direction_one, activations_unskewed_direction_two,
        #                                   activations_unskewed_direction_three, activations_unskewed_direction_four), 1)
        #
        # result = activations_combined
        # return result

        # activations_unskewed = self.compute_multi_dimensional_lstm(self.mdlstm_parameters,
        #                                                           x)
        activations_unskewed = self.compute_leaky_lp_cell(self.mdlstm_parameters,
                                                                   x)

        # print("len(activations_unskewed: " + str(len(activations_unskewed)))
        # print("activations_unskewed.size(): " + str(activations_unskewed.size()))

        result = MDLSTMExamplesPacking.extract_flipped_back_activations_from_unskewed_activations(activations_unskewed)

        # print("multi_dimensional_lstm - Time used for network forward: "
        #       + str(util.timing.milliseconds_since(time_start_network_forward)))

        return result

    @staticmethod
    def compute_weighted_input_input_gate(input_gate_input_column, mdlstm_parameters):
        input_gate_hidden_state_column = mdlstm_parameters.get_input_gate_hidden_state_column()
        input_gate_memory_state_column = mdlstm_parameters.get_input_gate_memory_state_column()
        input_gate_weighted_states_plus_weighted_input = input_gate_input_column + \
            input_gate_hidden_state_column + input_gate_memory_state_column
        return input_gate_weighted_states_plus_weighted_input

    def compute_weighted_input_output_gate(self, mdlstm_parameters,
                                           previous_memory_state_column,
                                           output_gate_input_column):

        if self.use_dropout:
            output_gate_memory_state_column = \
                F.dropout(mdlstm_parameters.
                          compute_output_gate_memory_state_weighted_input(previous_memory_state_column),
                          p=0.2, training=self.training)
        else:
            output_gate_memory_state_column = \
                mdlstm_parameters.compute_output_gate_memory_state_weighted_input(previous_memory_state_column)

        if self.clamp_gradients:
            output_gate_memory_state_column = InsideModelGradientClamping.\
                register_gradient_clamping_default_clamping_bound(
                output_gate_memory_state_column,
                "mdlstm - output_gate_memory_state_column")

        return self.compute_weighted_input_forget_gate(
                mdlstm_parameters.get_output_gate_hidden_state_column(),
                output_gate_memory_state_column,
                output_gate_input_column)

    @staticmethod
    def compute_weighted_input_forget_gate(forget_gate_hidden_state_column,
                                           forget_gate_memory_state_column,
                                           forget_gate_input_column):

        forget_gate_weighted_states_plus_weighted_input = forget_gate_input_column + forget_gate_hidden_state_column + \
            forget_gate_memory_state_column

        # print("forget_gate_memory_state_column: " + str(forget_gate_memory_state_column))
        # print("forget_gate_hidden_state_column: " + str(forget_gate_hidden_state_column))
        # print("forget_gate_input_column: " + str(forget_gate_input_column))

        # print("forget_gate_weighted_states_plus_weighted_input: " + str(forget_gate_weighted_states_plus_weighted_input))

        return forget_gate_weighted_states_plus_weighted_input

    @staticmethod
    def compute_weighted_input_lambda_gate(lambda_gate_hidden_state_column,
                                           lambda_gate_memory_state_column_one,
                                           lambda_gate_memory_state_column_two,
                                           forget_gate_input_column):

        lambda_gate_weighted_states_plus_weighted_input = forget_gate_input_column + lambda_gate_hidden_state_column + \
                                                          lambda_gate_memory_state_column_one + \
                                                          lambda_gate_memory_state_column_two

        return lambda_gate_weighted_states_plus_weighted_input

    def forward_one_directional_multi_dimensional_lstm(self, x):
        activations_unskewed = self.compute_multi_dimensional_lstm(self.mdlstm_parameters,
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

        # Additional re-un-skewing step
        activations_re_skewed_re_unskewed = ImageInputTransformer.\
            extract_unskewed_activations_from_activation_tensor(activations_re_skewed,
                                                                original_image_columns)
        # print("activations_re_skewed_re_unskewed.size(): " + str(activations_re_skewed_re_unskewed.size()))

        return activations_re_skewed_re_unskewed
        # return activations_unskewed

    # Needs to be implemented in the subclasses
    def _compute_multi_dimensional_function_one_direction(self, function_input):
        return self.compute_multi_dimensional_lstm(self.mdlstm_parameters, function_input)

    # Input tensor x is a batch of image tensors
    def forward(self, x):
        if self.compute_multi_directional_flag:
            # With distinct parameters for every direction
            return self.forward_multi_directional_multi_dimensional_lstm(x)
            # With same paramters for every direction
            #return self.forward_multi_directional_multi_dimensional_function_fast(x)
        else:
            # print(">>> Execute forward_one_directional_multi_dimensional_lstm")
            return self.forward_one_directional_multi_dimensional_lstm(x)
            #return self.\
            #    forward_one_directional_multi_dimensional_lstm_with_additional_skewing_unskewing_step(x)

    def set_use_examples_packing(self, use_examples_packing):
        self.use_example_packing = use_examples_packing
