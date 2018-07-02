from torch.nn.modules.module import Module
from modules.block_multi_dimensional_lstm_layer_pair import BlockMultiDimensionalLSTMLayerPair
from modules.block_multi_dimensional_lstm_layer_pair import BlockMultiDimensionalLSTM
from modules.block_strided_convolution import BlockStridedConvolution
import torch.nn as nn
from modules.size_two_dimensional import SizeTwoDimensional


class LayerPairSpecificParameters:
    def __init__(self, input_channels: int, mdlstm_hidden_states_size: int,
                 output_channels: int, mdlstm_block_size: SizeTwoDimensional,
                 block_strided_convolution_block_size: SizeTwoDimensional):
        self.input_channels = input_channels
        self.mdlstm_hidden_states_size = mdlstm_hidden_states_size
        self.output_channels = output_channels
        self.mdlstm_block_size = mdlstm_block_size
        self.block_strided_convolution_block_size = block_strided_convolution_block_size

    @staticmethod
    def create_layer_pair_specific_parameters(input_channels: int, mdlstm_hidden_states_size: int,
                                              output_channels: int, mdlstm_block_size: SizeTwoDimensional,
                                              block_strided_convolution_block_size: SizeTwoDimensional):
        return LayerPairSpecificParameters(input_channels, mdlstm_hidden_states_size,
                                           output_channels, mdlstm_block_size,
                                           block_strided_convolution_block_size)


class BlockMultiDimensionalLSTMLayerPairStacking(Module):

    def __init__(self, block_multi_dimensional_lstm_layer_pairs):
        super(BlockMultiDimensionalLSTMLayerPairStacking, self).__init__()
        # A module list is used to assure the variable number of layers is properly registered
        # so that things will be put on the right GPUs. It is necessary to "only"
        # use this module list and loop over the elements in this list in the forward method,
        # previously storing in a normal list in addition to a module list, and reading
        # from the normal list in the forward function caused things to still be put on
        # different GPUs when using more than one GPU
        self.block_multi_dimensional_lstm_layer_pairs = nn.ModuleList([])
        self.block_multi_dimensional_lstm_layer_pairs.extend(block_multi_dimensional_lstm_layer_pairs)

        print("len(self.block_multi_dimensional_lstm_layer_pairs): " + str(len(self.block_multi_dimensional_lstm_layer_pairs)))

    @staticmethod
    def create_block_multi_dimensional_lstm_pair_stacking(layer_pair_specific_parameters_list: list,
                                                          compute_multi_directional: bool,
                                                          clamp_gradients: bool,
                                                          use_dropout: bool,
                                                          input_and_output_are_lists: bool,
                                                          nonlinearity="tanh"
                                                          ):
        block_multi_dimensional_lstm_layer_pairs = list([])
        for layer_pair_specific_parameters in layer_pair_specific_parameters_list:
            layer_pair = BlockMultiDimensionalLSTMLayerPairStacking.\
                create_block_multi_dimensional_lstm_layer_pair(layer_pair_specific_parameters,
                                                               compute_multi_directional,
                                                               clamp_gradients,
                                                               use_dropout,
                                                               input_and_output_are_lists,
                                                               nonlinearity)
            block_multi_dimensional_lstm_layer_pairs.append(layer_pair)
        return BlockMultiDimensionalLSTMLayerPairStacking(block_multi_dimensional_lstm_layer_pairs)

    @staticmethod
    def create_block_multi_dimensional_lstm_layer_pair(layer_pair_specific_parameters: LayerPairSpecificParameters,
                                                       compute_multi_directional: bool,
                                                       clamp_gradients: bool,
                                                       use_dropout: bool,
                                                       input_and_output_are_lists,
                                                       nonlinearity="tanh"):
        return BlockMultiDimensionalLSTMLayerPair.create_block_multi_dimensional_lstm_layer_pair(
            layer_pair_specific_parameters.input_channels, layer_pair_specific_parameters.mdlstm_hidden_states_size,
            layer_pair_specific_parameters.output_channels, layer_pair_specific_parameters.mdlstm_block_size,
            layer_pair_specific_parameters.block_strided_convolution_block_size, compute_multi_directional,
            clamp_gradients,
            use_dropout, input_and_output_are_lists, nonlinearity)

    # This is an example of a network that stacks two
    # BlockMultiDimensionalLSTMLayerPair layers. It illustrates how
    # the block dimensions can be varied within layer pairs and across layer pairs.
    @staticmethod
    def create_two_layer_pair_network(first_mdlstm_hidden_states_size: int,
                                      mdlstm_block_size: SizeTwoDimensional,
                                      block_strided_convolution_block_size: SizeTwoDimensional,
                                      clamp_gradients: bool):
        compute_multi_directional = False
        use_dropout = False
        nonlinearity = "tanh"

        # Layer pair one
        input_channels = 1
        number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
                                              block_strided_convolution_block_size.height
        parameter_increase_factor = number_of_elements_reduction_factor
        output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        second_mdlstm_hidden_states_size = 4 * first_mdlstm_hidden_states_size
        output_channels = number_of_elements_reduction_factor * output_channels

        pair_two_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, second_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list = list([pair_one_specific_parameters, pair_two_specific_parameters])
        return BlockMultiDimensionalLSTMLayerPairStacking.\
            create_block_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                              compute_multi_directional, clamp_gradients, use_dropout,
                                                              nonlinearity)

        # This is an example of a network that stacks two
        # BlockMultiDimensionalLSTMLayerPair layers. It illustrates how
        # the block dimensions can be varied within layer pairs and across layer pairs.

    @staticmethod
    def create_three_layer_pair_network(first_mdlstm_hidden_states_size: int,
                                        mdlstm_block_size: SizeTwoDimensional,
                                        block_strided_convolution_block_size: SizeTwoDimensional,
                                        clamp_gradients, input_and_output_are_lists: bool):
        compute_multi_directional = False
        use_dropout = False
        nonlinearity = "tanh"

        # Layer pair one
        input_channels = 1
        number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
                                              block_strided_convolution_block_size.height
        parameter_increase_factor = number_of_elements_reduction_factor
        output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        second_mdlstm_hidden_states_size = 4 * first_mdlstm_hidden_states_size
        output_channels = number_of_elements_reduction_factor * output_channels

        pair_two_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, second_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair three
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        third_mdlstm_hidden_states_size = 4 * second_mdlstm_hidden_states_size
        output_channels = number_of_elements_reduction_factor * output_channels

        pair_three_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, third_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list = list([pair_one_specific_parameters, pair_two_specific_parameters,
                                                     pair_three_specific_parameters])
        return BlockMultiDimensionalLSTMLayerPairStacking. \
            create_block_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                              compute_multi_directional, clamp_gradients,
                                                              use_dropout, input_and_output_are_lists, nonlinearity)

    # Following "Handwriting Recognition with Large Multidimensional Long Short-Term
    # Memory Recurrent Neural Networks" (Voigtlander et.al, 2016)
    # Both the hidden states sizes and the output channels (which are like the hidden states
    # for the convolution layers) are now made into a linear factor of the layer number
    @staticmethod
    def create_three_layer_pair_network_linear_parameter_size_increase(
            input_channels, first_mdlstm_hidden_states_size: int,
            mdlstm_block_size: SizeTwoDimensional, block_strided_convolution_block_size: SizeTwoDimensional,
            compute_multi_directional: bool,
            clamp_gradients: bool, use_dropout: bool,
            input_and_output_are_lists: bool):

        nonlinearity = "tanh"

        # Layer pair one
        #number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
        #                                      block_strided_convolution_block_size.height
        # output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size
        output_channels = 2 * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with a factor that is
        # equal to the layer number
        second_mdlstm_hidden_states_size = 3 * first_mdlstm_hidden_states_size
        # output_channels = number_of_elements_reduction_factor * output_channels
        output_channels = 4 * first_mdlstm_hidden_states_size

        pair_two_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, second_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair three
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with a factor that is
        # equal to the layer number
        third_mdlstm_hidden_states_size = 5 * first_mdlstm_hidden_states_size
        # output_channels = number_of_elements_reduction_factor * output_channels
        output_channels = 6 * first_mdlstm_hidden_states_size

        pair_three_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, third_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list = list([pair_one_specific_parameters, pair_two_specific_parameters,
                                                     pair_three_specific_parameters])
        return BlockMultiDimensionalLSTMLayerPairStacking. \
            create_block_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                              compute_multi_directional,
                                                              clamp_gradients,
                                                              use_dropout,
                                                              input_and_output_are_lists, nonlinearity)



    @staticmethod
    def create_one_layer_pair_plus_second_block_convolution_layer_network(first_mdlstm_hidden_states_size: int,
                                      mdlstm_block_size: SizeTwoDimensional,
                                      block_strided_convolution_block_size: SizeTwoDimensional,
                                      clamp_gradients:bool):
        compute_multi_directional = False
        use_dropout = False
        nonlinearity = "tanh"

        # Layer pair one
        input_channels = 1
        number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
                                              block_strided_convolution_block_size.height
        output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        output_channels = number_of_elements_reduction_factor * output_channels

        first_layer = BlockMultiDimensionalLSTMLayerPairStacking.\
            create_block_multi_dimensional_lstm_layer_pair(pair_one_specific_parameters,
                                                           compute_multi_directional, use_dropout,
                                                           nonlinearity)
        second_block_convolution = BlockStridedConvolution.\
            create_block_strided_convolution(input_channels, output_channels,
                                             block_strided_convolution_block_size, clamp_gradients)
        layers = list([first_layer, second_block_convolution])
        return BlockMultiDimensionalLSTMLayerPairStacking(layers)

    @staticmethod
    def create_one_layer_pair_plus_second_block_mdlstm_layer_network(first_mdlstm_hidden_states_size: int,
                                                                          mdlstm_block_size: SizeTwoDimensional,
                                                                          block_strided_convolution_block_size: SizeTwoDimensional):
        compute_multi_directional = False
        use_dropout = False
        nonlinearity = "tanh"

        # Layer pair one
        input_channels = 1
        number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
                                              block_strided_convolution_block_size.height
        output_channels = 4 * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution

        first_layer = BlockMultiDimensionalLSTMLayerPairStacking. \
            create_block_multi_dimensional_lstm_layer_pair(pair_one_specific_parameters,
                                                           compute_multi_directional, use_dropout,
                                                           nonlinearity)
        second_mdlstm_hidden_states_size = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size
        second_block_mdlstm = BlockMultiDimensionalLSTM.\
            create_block_multi_dimensional_lstm(input_channels, second_mdlstm_hidden_states_size,
                                                mdlstm_block_size, compute_multi_directional, use_dropout,
                                                           nonlinearity)
        layers = list([first_layer, second_block_mdlstm])
        return BlockMultiDimensionalLSTMLayerPairStacking(layers)

    @staticmethod
    def create_module_list(block_multi_dimensional_lstm_layer_pairs):
        module_list = list([])
        for layer_pair in block_multi_dimensional_lstm_layer_pairs:
            module_list.append(layer_pair)
            module_list.extend(layer_pair.module_list)
        return module_list

    def set_training(self, training):
        for layer_pair in self.block_multi_dimensional_lstm_layer_pairs:
            layer_pair.set_training(training)

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        layer_input_size = input_size
        for layer_pair in self.block_multi_dimensional_lstm_layer_pairs:
            print("layer_input_size: " + str(layer_input_size))
            result = layer_pair.get_number_of_output_dimensions(layer_input_size)
            layer_input_size = layer_pair.get_output_size_two_dimensional(layer_input_size)
            print("new layer_input_size: " + str(layer_input_size))
            print("number of output dimensions layer: " + str(result))
        return result

    def get_number_of_output_channels(self):
        last_layer_pair = self.block_multi_dimensional_lstm_layer_pairs[
            len(self.block_multi_dimensional_lstm_layer_pairs) - 1]
        return last_layer_pair.get_number_of_output_channels()

    def get_width_reduction_factor(self):
        result = 1
        for layer_pair in self.block_multi_dimensional_lstm_layer_pairs:
            result *= layer_pair.block_strided_convolution.get_width_reduction_factor()

        return result

    def get_height_reduction_factor(self):
        result = 1
        for layer_pair in self.block_multi_dimensional_lstm_layer_pairs:
            result *= layer_pair.block_strided_convolution.get_height_reduction_factor()

        return result

    def forward(self, x):
        network_input = x
        for layer_pair in self.block_multi_dimensional_lstm_layer_pairs:
            output = layer_pair(network_input)
            # print(">>> BlockMultiDimensionalLSTMLayerPairStacking.forward: - output.grad_fn "
            #      + str(output.grad_fn))
            # print("output.size(): " + str(output.size()))
            network_input = output
        return output
