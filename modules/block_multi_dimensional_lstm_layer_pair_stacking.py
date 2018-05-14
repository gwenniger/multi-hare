from torch.nn.modules.module import Module
from modules.block_multi_dimensional_lstm_layer_pair import BlockMultiDimensionalLSTMLayerPair
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
        self.block_multi_dimensional_lstm_layer_pairs = block_multi_dimensional_lstm_layer_pairs
        # A module list is used to assure the variable number of layers is properly registered
        # so that things will be put on the right GPUs
        self.module_list = BlockMultiDimensionalLSTMLayerPairStacking.\
            create_module_list(block_multi_dimensional_lstm_layer_pairs)

    @staticmethod
    def create_block_multi_dimensional_lstm_pair_stacking(layer_pair_specific_parameters_list: list,
                                                          compute_multi_directional: bool, use_dropout: bool,
                                                          nonlinearity="tanh"
                                                          ):
        block_multi_dimensional_lstm_layer_pairs = list([])
        for layer_pair_specific_parameters in layer_pair_specific_parameters_list:
            layer_pair = BlockMultiDimensionalLSTMLayerPairStacking.\
                create_block_multi_dimensional_lstm_layer_pair(layer_pair_specific_parameters,
                                                               compute_multi_directional,
                                                               use_dropout,
                                                               nonlinearity)
            block_multi_dimensional_lstm_layer_pairs.append(layer_pair)
        return BlockMultiDimensionalLSTMLayerPairStacking(block_multi_dimensional_lstm_layer_pairs)

    @staticmethod
    def create_block_multi_dimensional_lstm_layer_pair(layer_pair_specific_parameters: LayerPairSpecificParameters,
                                                       compute_multi_directional: bool, use_dropout: bool,
                                                       nonlinearity="tanh"):
        return BlockMultiDimensionalLSTMLayerPair.create_block_multi_dimensional_lstm_layer_pair(
            layer_pair_specific_parameters.input_channels, layer_pair_specific_parameters.mdlstm_hidden_states_size,
            layer_pair_specific_parameters.output_channels, layer_pair_specific_parameters.mdlstm_block_size,
            layer_pair_specific_parameters.block_strided_convolution_block_size, compute_multi_directional,
            use_dropout, nonlinearity)

    # This is an example of a network that stacks two
    # BlockMultiDimensionalLSTMLayerPair layers. It illustrates how
    # the block dimensions can be varied within layer pairs and across layer pairs.
    @staticmethod
    def create_two_layer_pair_network(first_mdlstm_hidden_states_size: int,
                                      mdlstm_block_size: SizeTwoDimensional,
                                      block_strided_convolution_block_size: SizeTwoDimensional):
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
        second_mdlstm_hidden_states_size = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size
        output_channels = number_of_elements_reduction_factor * output_channels

        pair_two_specific_parameters = LayerPairSpecificParameters.create_layer_pair_specific_parameters(
            input_channels, second_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list = list([pair_one_specific_parameters, pair_two_specific_parameters])
        return BlockMultiDimensionalLSTMLayerPairStacking.\
            create_block_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                              compute_multi_directional, use_dropout, nonlinearity)
    @staticmethod
    def create_module_list(block_multi_dimensional_lstm_layer_pairs):
        module_list = nn.ModuleList([])
        for layer_pair in block_multi_dimensional_lstm_layer_pairs:
            module_list.append(layer_pair)
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
            print("number of output dimensions layer: " + str(result))
        return result

    def forward(self, x):
        output = x
        for layer_pair in self.block_multi_dimensional_lstm_layer_pairs:
            output = layer_pair(output)
            # print(">>> BlockMultiDimensionalLSTMLayerPairStacking.forward: - output.grad_fn "
            #      + str(output.grad_fn))
            # print("output.size(): " + str(output.size()))
        return output
