from torch.nn.modules.module import Module
from modules.mdlstm_layer_block_strided_convolution_layer_pair import MDLSTMLayerBlockStridedConvolutionLayerPair
from modules.mdlstm_layer_block_strided_convolution_layer_pair import BlockMultiDimensionalLSTM
from modules.block_strided_convolution import BlockStridedConvolution
import torch.nn as nn
from modules.size_two_dimensional import SizeTwoDimensional
from modules.multi_dimensional_lstm import MultiDimensionalLSTM


class MDLSTMLayerPairSpecificParameters:
    def __init__(self, input_channels: int, mdlstm_hidden_states_size: int,
                 output_channels: int, block_strided_convolution_block_size: SizeTwoDimensional,
                 block_strided_convolution_shares_weights_across_directions: bool = False):
        self.input_channels = input_channels
        self.mdlstm_hidden_states_size = mdlstm_hidden_states_size
        self.output_channels = output_channels
        self.block_strided_convolution_block_size = block_strided_convolution_block_size
        self.block_strided_convolution_shares_weights_across_directions = \
            block_strided_convolution_shares_weights_across_directions


    @staticmethod
    def create_mdlstm_layer_pair_specific_parameters(input_channels: int, mdlstm_hidden_states_size: int,
                                                     output_channels: int,
                                                     block_strided_convolution_block_size: SizeTwoDimensional,
                                                     block_strided_convolution_shares_weights_across_directions: bool):
        return MDLSTMLayerPairSpecificParameters(input_channels, mdlstm_hidden_states_size,
                                                 output_channels, block_strided_convolution_block_size,
                                                 block_strided_convolution_shares_weights_across_directions)


class BlockMDLSTMLayerPairSpecificParameters(MDLSTMLayerPairSpecificParameters):
    def __init__(self, input_channels: int, mdlstm_hidden_states_size: int,
                 output_channels: int, block_strided_convolution_block_size: SizeTwoDimensional,
                 mdlstm_block_size: SizeTwoDimensional,
                 block_strided_convolution_shares_weights_across_directions: bool):
        super(BlockMDLSTMLayerPairSpecificParameters, self).\
            __init__(input_channels, mdlstm_hidden_states_size, output_channels,
                     block_strided_convolution_block_size, block_strided_convolution_shares_weights_across_directions)
        self.mdlstm_block_size = mdlstm_block_size

    @staticmethod
    def create_block_mdlstm_layer_pair_specific_parameters(input_channels: int, mdlstm_hidden_states_size: int,
                                                           output_channels: int, mdlstm_block_size: SizeTwoDimensional,
                                                           block_strided_convolution_block_size: SizeTwoDimensional,
                                                           block_strided_convolution_shares_weights_across_directions:
                                                           bool):
        return BlockMDLSTMLayerPairSpecificParameters(input_channels, mdlstm_hidden_states_size,
                                                      output_channels, mdlstm_block_size,
                                                      block_strided_convolution_block_size,
                                                      block_strided_convolution_shares_weights_across_directions)


class MultiDimensionalLSTMLayerPairStacking(Module):

    def __init__(self, block_multi_dimensional_lstm_layer_pairs):
        super(MultiDimensionalLSTMLayerPairStacking, self).__init__()
        # A module list is used to assure the variable number of layers is properly registered
        # so that things will be put on the right GPUs. It is necessary to "only"
        # use this module list and loop over the elements in this list in the forward method,
        # previously storing in a normal list in addition to a module list, and reading
        # from the normal list in the forward function caused things to still be put on
        # different GPUs when using more than one GPU
        self.multi_dimensional_lstm_layer_pairs = nn.ModuleList([])
        self.multi_dimensional_lstm_layer_pairs.extend(block_multi_dimensional_lstm_layer_pairs)

        print("len(self.block_multi_dimensional_lstm_layer_pairs): " + str(len(self.multi_dimensional_lstm_layer_pairs)))

    @staticmethod
    def create_block_multi_dimensional_lstm_pair_stacking(layer_pair_specific_parameters_list: list,
                                                          compute_multi_directional: bool,
                                                          clamp_gradients: bool,
                                                          use_dropout: bool,
                                                          use_bias_with_block_strided_convolution: bool,
                                                          nonlinearity="tanh"
                                                          ):
        block_multi_dimensional_lstm_layer_pairs = list([])
        layer_pair_index = 0
        for layer_pair_specific_parameters in layer_pair_specific_parameters_list:
            layer_pair = MultiDimensionalLSTMLayerPairStacking.\
                create_block_multi_dimensional_lstm_layer_pair(layer_pair_index,
                                                               layer_pair_specific_parameters,
                                                               compute_multi_directional,
                                                               clamp_gradients,
                                                               use_dropout,
                                                               use_bias_with_block_strided_convolution,
                                                               nonlinearity)
            block_multi_dimensional_lstm_layer_pairs.append(layer_pair)
            layer_pair_index += 1
        return MultiDimensionalLSTMLayerPairStacking(block_multi_dimensional_lstm_layer_pairs)

    @staticmethod
    def create_multi_dimensional_lstm_layer_pairs_list(
            layer_pair_specific_parameters_list: list, compute_multi_directional: bool,
            clamp_gradients: bool, use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            use_example_packing: bool,
            use_leaky_lp_cells: bool,
            nonlinearity="tanh"):
        multi_dimensional_lstm_layer_pairs = list([])
        layer_pair_index = 0
        for layer_pair_specific_parameters in layer_pair_specific_parameters_list:
            print("layer_pair_specific_parameters: " + str(layer_pair_specific_parameters))
            layer_pair = MultiDimensionalLSTMLayerPairStacking. \
                create_multi_dimensional_lstm_layer_pair(layer_pair_index, layer_pair_specific_parameters,
                                                         compute_multi_directional,
                                                         clamp_gradients,
                                                         use_dropout,
                                                         use_bias_with_block_strided_convolution,
                                                         use_example_packing,
                                                         use_leaky_lp_cells,
                                                         nonlinearity)
            multi_dimensional_lstm_layer_pairs.append(layer_pair)
            layer_pair_index += 1
        return multi_dimensional_lstm_layer_pairs

    @staticmethod
    def create_multi_dimensional_lstm_pair_stacking(layer_pair_specific_parameters_list: list,
                                                    compute_multi_directional: bool,
                                                    clamp_gradients: bool,
                                                    use_dropout: bool,
                                                    use_bias_with_block_strided_convolution: bool,
                                                    use_example_packing: bool,
                                                    use_leaky_lp_cells: bool,
                                                    nonlinearity="tanh"
                                                    ):
        multi_dimensional_lstm_layer_pairs = MultiDimensionalLSTMLayerPairStacking.\
            create_multi_dimensional_lstm_layer_pairs_list(
                layer_pair_specific_parameters_list, compute_multi_directional,
                clamp_gradients, use_dropout, use_bias_with_block_strided_convolution,
                use_example_packing, use_leaky_lp_cells, nonlinearity)
        return MultiDimensionalLSTMLayerPairStacking(multi_dimensional_lstm_layer_pairs)

    @staticmethod
    def create_block_multi_dimensional_lstm_layer_pair(
            layer_pair_specific_parameters: BlockMDLSTMLayerPairSpecificParameters,
            compute_multi_directional: bool,
            clamp_gradients: bool,
            use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            nonlinearity="tanh"):

        return MDLSTMLayerBlockStridedConvolutionLayerPair.create_block_mdlstm_block_strided_convolution_layer_pair(
            layer_pair_specific_parameters.input_channels, layer_pair_specific_parameters.mdlstm_hidden_states_size,
            layer_pair_specific_parameters.output_channels, layer_pair_specific_parameters.mdlstm_block_size,
            layer_pair_specific_parameters.block_strided_convolution_block_size, compute_multi_directional,
            clamp_gradients,
            use_dropout,
            use_bias_with_block_strided_convolution,
            nonlinearity)

    @staticmethod
    def create_multi_dimensional_lstm_layer_pair(
            layer_index,
            layer_pair_specific_parameters: MDLSTMLayerPairSpecificParameters,
            compute_multi_directional: bool,
            clamp_gradients: bool,
            use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            use_example_packing: bool,
            use_leaky_lp_cells: bool,
            nonlinearity="tanh"):

        return MDLSTMLayerBlockStridedConvolutionLayerPair.create_mdlstm_block_strided_convolution_layer_pair(
            layer_index,
            layer_pair_specific_parameters.input_channels, layer_pair_specific_parameters.mdlstm_hidden_states_size,
            layer_pair_specific_parameters.output_channels,
            layer_pair_specific_parameters.block_strided_convolution_block_size, compute_multi_directional,
            clamp_gradients,
            use_dropout,
            use_bias_with_block_strided_convolution,
            use_example_packing,
            use_leaky_lp_cells,
            layer_pair_specific_parameters.block_strided_convolution_shares_weights_across_directions,
            nonlinearity)

    # This is an example of a network that stacks two
    # BlockMultiDimensionalLSTMLayerPair layers. It illustrates how
    # the block dimensions can be varied within layer pairs and across layer pairs.
    @staticmethod
    def create_two_layer_pair_network(first_mdlstm_hidden_states_size: int,
                                      mdlstm_block_size: SizeTwoDimensional,
                                      block_strided_convolution_block_size: SizeTwoDimensional,
                                      compute_multi_directional,
                                      clamp_gradients: bool,
                                      use_bias_with_block_strided_convolution: bool,
                                      use_example_packing: bool,
                                      use_leaky_lp_cells: bool):
        use_dropout = False
        nonlinearity = "tanh"

        # Layer pair one
        input_channels = 1
        number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
                                              block_strided_convolution_block_size.height
        parameter_increase_factor = number_of_elements_reduction_factor
        output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = BlockMDLSTMLayerPairSpecificParameters.create_block_mdlstm_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        second_mdlstm_hidden_states_size = 4 * first_mdlstm_hidden_states_size
        output_channels = number_of_elements_reduction_factor * output_channels

        pair_two_specific_parameters = BlockMDLSTMLayerPairSpecificParameters.create_block_mdlstm_layer_pair_specific_parameters(
            input_channels, second_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list = list([pair_one_specific_parameters, pair_two_specific_parameters])
        return MultiDimensionalLSTMLayerPairStacking.\
            create_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                        compute_multi_directional, clamp_gradients, use_dropout,
                                                        use_bias_with_block_strided_convolution,
                                                        use_example_packing,
                                                        use_leaky_lp_cells,
                                                        nonlinearity)

        # This is an example of a network that stacks two
        # BlockMultiDimensionalLSTMLayerPair layers. It illustrates how
        # the block dimensions can be varied within layer pairs and across layer pairs.

    @staticmethod
    def create_three_layer_pair_network(first_mdlstm_hidden_states_size: int,
                                        mdlstm_block_size: SizeTwoDimensional,
                                        block_strided_convolution_block_size: SizeTwoDimensional,
                                        clamp_gradients):
        compute_multi_directional = False
        use_dropout = False
        nonlinearity = "tanh"

        # Layer pair one
        input_channels = 1
        number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
                                              block_strided_convolution_block_size.height
        parameter_increase_factor = number_of_elements_reduction_factor
        output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = BlockMDLSTMLayerPairSpecificParameters.create_block_mdlstm_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        second_mdlstm_hidden_states_size = 4 * first_mdlstm_hidden_states_size
        output_channels = number_of_elements_reduction_factor * output_channels

        pair_two_specific_parameters = BlockMDLSTMLayerPairSpecificParameters.create_block_mdlstm_layer_pair_specific_parameters(
            input_channels, second_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair three
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        third_mdlstm_hidden_states_size = 4 * second_mdlstm_hidden_states_size
        output_channels = number_of_elements_reduction_factor * output_channels

        pair_three_specific_parameters = BlockMDLSTMLayerPairSpecificParameters.create_block_mdlstm_layer_pair_specific_parameters(
            input_channels, third_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list = list([pair_one_specific_parameters, pair_two_specific_parameters,
                                                     pair_three_specific_parameters])
        return MultiDimensionalLSTMLayerPairStacking. \
            create_block_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                              compute_multi_directional, clamp_gradients,
                                                              use_dropout, nonlinearity)


    @staticmethod
    def block_mdlstm_parameter_creation_function(
            input_channels, mdlstm_hidden_states_size, output_channels,
            mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            block_strided_convolution_shares_weights_across_directions: bool):

        return BlockMDLSTMLayerPairSpecificParameters.create_block_mdlstm_layer_pair_specific_parameters(
            input_channels, mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size,
            block_strided_convolution_shares_weights_across_directions)

    @staticmethod
    def mdlstm_parameter_creation_function(
            input_channels,  mdlstm_hidden_states_size, output_channels,
            mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            block_strided_convolution_shares_weights_across_directions: bool):
        return MDLSTMLayerPairSpecificParameters.create_mdlstm_layer_pair_specific_parameters(
            input_channels, mdlstm_hidden_states_size, output_channels, block_strided_convolution_block_size,
            block_strided_convolution_shares_weights_across_directions)

    # Following "Handwriting Recognition with Large Multidimensional Long Short-Term
    # Memory Recurrent Neural Networks" (Voigtlander et.al, 2016)
    # Both the hidden states sizes and the output channels (which are like the hidden states
    # for the convolution layers) are now made into a linear factor of the layer number
    # The parameter_creation_function controls whether block_mdlstm parameters
    # or mdlstm parameters are created in the final result list
    @staticmethod
    def create_three_layer_pair_network_linear_parameter_size_increase_parameters(
            input_channels,
            first_mdlstm_hidden_states_size: int,
            mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            parameter_creation_function):

        # Layer pair one
        # number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
        #                                      block_strided_convolution_block_size.height
        # output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size
        output_channels = 2 * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = parameter_creation_function(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with a factor that is
        # equal to the layer number
        second_mdlstm_hidden_states_size = 3 * first_mdlstm_hidden_states_size
        # output_channels = number_of_elements_reduction_factor * output_channels
        output_channels = 4 * first_mdlstm_hidden_states_size

        pair_two_specific_parameters = parameter_creation_function(
            input_channels, second_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair three
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with a factor that is
        # equal to the layer number
        third_mdlstm_hidden_states_size = 5 * first_mdlstm_hidden_states_size
        # output_channels = number_of_elements_reduction_factor * output_channels
        output_channels = 6 * first_mdlstm_hidden_states_size

        pair_three_specific_parameters = parameter_creation_function(
            input_channels, third_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list = list([pair_one_specific_parameters, pair_two_specific_parameters,
                                                     pair_three_specific_parameters])
        return layer_pairs_specific_parameters_list


    @staticmethod
    def share_weights_for_block_strided_convolution_layer_with_index(
            block_strided_convolution_layers_using_weight_sharing: list, index):
        """
        Determine whether the block-strided convolution weights are to be shared
        for the block-strided convolution layer with given index
        :param block_strided_convolution_layers_using_weight_sharing: A list of indices of
        block-strided convolution layers that should use weight sharing
        :param index: the index of the layer for which weight-sharing or not is to be determined

        :return: A boolean that specifies using weight sharing or not for the block-strided
        convolution layer with this index
        """
        return index in block_strided_convolution_layers_using_weight_sharing

    @staticmethod
    def create_first_two_layer_pair_parameters_with_two_channels_per_direction_first_mdlstm_layer(
            input_channels,
            mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            mdlstm_layer_sizes: list,
            parameter_creation_function,
            block_strided_convolution_layers_using_weight_sharing: list,
            use_dropout: bool):

        # Layer pair one
        # number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
        #                                      block_strided_convolution_block_size.height
        # output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size_per_direction

        first_mdlstm_hidden_states_size_per_direction = mdlstm_layer_sizes[0]
        print("first_mdlstm_hidden_states_size_per_direction: " + str(first_mdlstm_hidden_states_size_per_direction))

        output_channels = 6

        pair_one_specific_parameters = parameter_creation_function(
            input_channels, first_mdlstm_hidden_states_size_per_direction, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size,
            MultiDimensionalLSTMLayerPairStacking.share_weights_for_block_strided_convolution_layer_with_index(
                block_strided_convolution_layers_using_weight_sharing, 0))

        # Layer pair two
        input_channels = output_channels
        second_mdlstm_hidden_states_size_per_direction = mdlstm_layer_sizes[1]
        print("second_mdlstm_hidden_states_size_per_direction: " + str(second_mdlstm_hidden_states_size_per_direction))
        output_channels = 20

        pair_two_specific_parameters = parameter_creation_function(
            input_channels, second_mdlstm_hidden_states_size_per_direction,
            output_channels, mdlstm_block_size,
            block_strided_convolution_block_size,
            MultiDimensionalLSTMLayerPairStacking.share_weights_for_block_strided_convolution_layer_with_index(
                block_strided_convolution_layers_using_weight_sharing, 1)
        )

        result = list([pair_one_specific_parameters, pair_two_specific_parameters])
        return result

    """
    The structure and number of hidden units per layer are inspired on 
    "Dropout improves Recurrent Neural Networks for Handwriting Recognition"
    (Pham et.al, 2015). The crucial architectural choice of this network 
    is to start the first MDLSTM layers for the four directions with only 
    2 hidden units per direction.
    Since the resolution for the following layers decreases rapidly with 
    4 times 2 BlockStridedConvolution blocks, it is cheaper to have a higher 
    number of channels for the following layers, but expensive to have it 
    for the first layer. Note that this network is still not exactly identical 
    in structure to the one described in (Pham et.al, 2015), since in that 
    network the third BlockStridedConvolution layer is missing and replaced 
    by a fully connected layer . Ideally, both structures should be tested 
    to see if there's any difference in performance. 
    """
    @staticmethod
    def create_three_layer_pair_network_parameters_with_two_channels_per_direction_first_mdlstm_layer(
            input_channels,
            mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            parameter_creation_function,
            use_dropout: bool
    ):

        layer_pairs_specific_parameters_list = \
            MultiDimensionalLSTMLayerPairStacking.\
            create_first_two_layer_pair_parameters_with_two_channels_per_direction_first_mdlstm_layer(
                input_channels, mdlstm_block_size, block_strided_convolution_block_size, parameter_creation_function,
                use_dropout
            )

        # Layer pair three
        input_channels = 20
        # Here the number of mdstlm_hidden_states and output channels are increased with a factor that is
        # equal to the layer number
        # if use_dropout:
        #     third_mdlstm_hidden_states_size_per_direction = 100
        # else:
        # Use MDLSTM layers of size 2, 20, 50  with dropout in the second and third MDLSTM layer.
        # Doubling (also) the size of the third MDLSTM layer doesn't help for IAM, according to
        # "Dropout improves Recurrent Neural Networks for Handwriting Recognition"
        third_mdlstm_hidden_states_size_per_direction = 50
        output_channels = 50

        pair_three_specific_parameters = parameter_creation_function(
            input_channels, third_mdlstm_hidden_states_size_per_direction, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        layer_pairs_specific_parameters_list.append(pair_three_specific_parameters)
        return layer_pairs_specific_parameters_list


    # Creates a network of three layer pairs, each pair consisting of a block-MDLSTM layer
    # followed by a block-strided convolution layer.
    # This is the old version of the network that does not work well, presumably
    # because it is not a good idea to compute the MDLSTM features only within
    # blocks
    @staticmethod
    def create_block_mdlstm_three_layer_pair_network_linear_parameter_size_increase(
            input_channels, first_mdlstm_hidden_states_size: int,
            mdlstm_block_size: SizeTwoDimensional, block_strided_convolution_block_size: SizeTwoDimensional,
            compute_multi_directional: bool,
            clamp_gradients: bool, use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            use_example_packing: bool
    ):

        nonlinearity = "tanh"
        layer_pairs_specific_parameters_list = MultiDimensionalLSTMLayerPairStacking.\
            create_three_layer_pair_network_linear_parameter_size_increase_parameters(
                input_channels, first_mdlstm_hidden_states_size, mdlstm_block_size,
                block_strided_convolution_block_size,
                MultiDimensionalLSTMLayerPairStacking.block_mdlstm_parameter_creation_function)

        return MultiDimensionalLSTMLayerPairStacking. \
            create_block_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                              compute_multi_directional,
                                                              clamp_gradients,
                                                              use_dropout,
                                                              use_bias_with_block_strided_convolution,
                                                              use_example_packing,
                                                              nonlinearity)

    # Creates a network of three layer pairs, each pair consisting of a MDLSTM layer
    # followed by a block-strided convolution layer. This is the improved version
    # that follows the more typical architecture used for handwriting recognition.
    # Crucially the MDLSTM layers in this version of the network compute over the
    # entire layer and are not restricted to within-block computation
    @staticmethod
    def create_mdlstm_three_layer_pair_network_linear_parameter_size_increase(
            input_channels, first_mdlstm_hidden_states_size: int,
            block_strided_convolution_block_size: SizeTwoDimensional,
            compute_multi_directional: bool,
            clamp_gradients: bool, use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            use_example_packing: bool,
            use_leaky_lp_cells: bool):

        nonlinearity = "tanh"
        layer_pairs_specific_parameters_list = MultiDimensionalLSTMLayerPairStacking. \
            create_three_layer_pair_network_linear_parameter_size_increase_parameters(
                input_channels, first_mdlstm_hidden_states_size, None,
                block_strided_convolution_block_size,
                MultiDimensionalLSTMLayerPairStacking.mdlstm_parameter_creation_function)
        print("layers_pairs_specific_parameters_list: " + str(layer_pairs_specific_parameters_list))

        return MultiDimensionalLSTMLayerPairStacking. \
            create_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                        compute_multi_directional,
                                                        clamp_gradients,
                                                        use_dropout,
                                                        use_bias_with_block_strided_convolution,
                                                        use_example_packing,
                                                        use_leaky_lp_cells,
                                                        nonlinearity)

        # Creates a network of three layer pairs, each pair consisting of a MDLSTM layer
        # followed by a block-strided convolution layer. This is the improved version
        # that follows the more typical architecture used for handwriting recognition.
        # Crucially the MDLSTM layers in this version of the network compute over the
        # entire layer and are not restricted to within-block computation

    @staticmethod
    def create_mdlstm_three_layer_pair_network_with_two_channels_per_direction_first_mdlstm_layer(
            input_channels,
            block_strided_convolution_block_size: SizeTwoDimensional,
            compute_multi_directional: bool,
            clamp_gradients: bool, use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            use_example_packing: bool,
            use_leaky_lp_cells: bool):

        nonlinearity = "tanh"
        layer_pairs_specific_parameters_list = MultiDimensionalLSTMLayerPairStacking. \
            create_three_layer_pair_network_parameters_with_two_channels_per_direction_first_mdlstm_layer(
                input_channels, None,
                block_strided_convolution_block_size,
                MultiDimensionalLSTMLayerPairStacking.mdlstm_parameter_creation_function,
                use_dropout)
        print("layers_pairs_specific_parameters_list: " + str(layer_pairs_specific_parameters_list))

        return MultiDimensionalLSTMLayerPairStacking. \
            create_multi_dimensional_lstm_pair_stacking(layer_pairs_specific_parameters_list,
                                                        compute_multi_directional,
                                                        clamp_gradients,
                                                        use_dropout,
                                                        use_bias_with_block_strided_convolution,
                                                        use_example_packing,
                                                        use_leaky_lp_cells,
                                                        nonlinearity)

    @staticmethod
    def create_mdlstm_two_and_half_layer_pair_network_with_two_channels_per_direction_first_mdlstm_layer(
            input_channels,
            block_strided_convolution_block_size: SizeTwoDimensional,
            mdlstm_layer_sizes: list,
            compute_multi_directional: bool,
            clamp_gradients: bool, use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            use_example_packing: bool,
            use_leaky_lp_cells: bool,
            block_strided_convolution_layers_using_weight_sharing: list):

        """
        This function creates the first five layers of the
        network described in "Dropout improves Recurrent Neural
        Networks for Handwriting Recognition", which consists of
        two layer pairs of MDLSTM + BlockStridedConvolution and
        one third MDLSTM layer.

        The full network finishes with four fully connected layers that consume
        the output of the network created by this function, followed by summation
        and softmax. These final layers are not created here.

        :param input_channels:
        :param block_strided_convolution_block_size:
        :param compute_multi_directional:
        :param clamp_gradients:
        :param use_dropout:
        :param use_bias_with_block_strided_convolution:
        :param use_example_packing:
        :param use_leaky_lp_cells:
        :return:
        """

        nonlinearity = "tanh"
        layer_pairs_specific_parameters_list = MultiDimensionalLSTMLayerPairStacking. \
            create_first_two_layer_pair_parameters_with_two_channels_per_direction_first_mdlstm_layer(
                input_channels, None,
                block_strided_convolution_block_size,
                mdlstm_layer_sizes,
                MultiDimensionalLSTMLayerPairStacking.mdlstm_parameter_creation_function,
                block_strided_convolution_layers_using_weight_sharing,
                use_dropout)

        print("layers_pairs_specific_parameters_list: " + str(layer_pairs_specific_parameters_list))

        multi_dimensional_lstm_layer_pairs =  MultiDimensionalLSTMLayerPairStacking.\
            create_multi_dimensional_lstm_layer_pairs_list(
                layer_pairs_specific_parameters_list, compute_multi_directional,
                clamp_gradients, use_dropout, use_bias_with_block_strided_convolution,
                use_example_packing, use_leaky_lp_cells, nonlinearity)

        # Single Layer three (first two layers are layer pairs)
        layer_index = 2
        last_layer_pair = multi_dimensional_lstm_layer_pairs[-1]
        input_channels = last_layer_pair.get_number_of_output_channels()

        mdlstm_hidden_states_size = mdlstm_layer_sizes[2]

        print("third_mdlstm_hidden_states_size_per_direction: " + str(mdlstm_layer_sizes[2]))

        if compute_multi_directional:
            third_mdlstm_layer = MultiDimensionalLSTM. \
                create_multi_dimensional_lstm_fully_parallel(layer_index, input_channels, mdlstm_hidden_states_size,
                                                             compute_multi_directional,
                                                             clamp_gradients,
                                                             use_dropout,
                                                             use_example_packing,
                                                             use_leaky_lp_cells,
                                                             nonlinearity)
        else:
            third_mdlstm_layer = MultiDimensionalLSTM.create_multi_dimensional_lstm_fast(
                layer_index, input_channels, mdlstm_hidden_states_size,
                compute_multi_directional,
                clamp_gradients,
                use_dropout,
                use_example_packing,
                use_leaky_lp_cells,
                nonlinearity)

        multi_dimensional_lstm_layer_pairs.append(third_mdlstm_layer)

        return MultiDimensionalLSTMLayerPairStacking(multi_dimensional_lstm_layer_pairs)


    @staticmethod
    def create_one_layer_pair_plus_second_block_convolution_layer_network(
            first_mdlstm_hidden_states_size: int,
            mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            clamp_gradients:bool, use_bias_with_block_strided_convolution):

        compute_multi_directional = False
        use_dropout = False
        nonlinearity = "tanh"

        # Layer pair one
        input_channels = 1
        number_of_elements_reduction_factor = block_strided_convolution_block_size.width * \
                                              block_strided_convolution_block_size.height
        output_channels = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size

        pair_one_specific_parameters = BlockMDLSTMLayerPairSpecificParameters.create_block_mdlstm_layer_pair_specific_parameters(
            input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
            block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution
        output_channels = number_of_elements_reduction_factor * output_channels

        first_layer = MultiDimensionalLSTMLayerPairStacking.\
            create_block_multi_dimensional_lstm_layer_pair(pair_one_specific_parameters,
                                                           compute_multi_directional, use_dropout,
                                                           nonlinearity)
        second_block_convolution = BlockStridedConvolution.\
            create_block_strided_convolution(input_channels, output_channels,
                                             block_strided_convolution_block_size, clamp_gradients,
                                             use_bias_with_block_strided_convolution)
        layers = list([first_layer, second_block_convolution])
        return MultiDimensionalLSTMLayerPairStacking(layers)

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

        pair_one_specific_parameters = BlockMDLSTMLayerPairSpecificParameters.\
            create_block_mdlstm_layer_pair_specific_parameters(
                input_channels, first_mdlstm_hidden_states_size, output_channels, mdlstm_block_size,
                block_strided_convolution_block_size)

        # Layer pair two
        input_channels = output_channels
        # Here the number of mdstlm_hidden_states and output channels are increased with the
        # number of elements reduction factor from the dimensionality reduction with the
        # block_strided_convolution

        first_layer = MultiDimensionalLSTMLayerPairStacking. \
            create_block_multi_dimensional_lstm_layer_pair(pair_one_specific_parameters,
                                                           compute_multi_directional, use_dropout,
                                                           nonlinearity)
        second_mdlstm_hidden_states_size = number_of_elements_reduction_factor * first_mdlstm_hidden_states_size
        second_block_mdlstm = BlockMultiDimensionalLSTM.\
            create_block_multi_dimensional_lstm(input_channels, second_mdlstm_hidden_states_size,
                                                mdlstm_block_size, compute_multi_directional, use_dropout,
                                                           nonlinearity)
        layers = list([first_layer, second_block_mdlstm])
        return MultiDimensionalLSTMLayerPairStacking(layers)

    @staticmethod
    def create_module_list(block_multi_dimensional_lstm_layer_pairs):
        module_list = list([])
        for layer_pair in block_multi_dimensional_lstm_layer_pairs:
            module_list.append(layer_pair)
            module_list.extend(layer_pair.module_list)
        return module_list

    def set_training(self, training):
        for layer_pair in self.multi_dimensional_lstm_layer_pairs:
            layer_pair.set_training(training)

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        layer_input_size = input_size
        for layer_pair in self.multi_dimensional_lstm_layer_pairs:
            print("layer_input_size: " + str(layer_input_size))
            result = layer_pair.get_number_of_output_dimensions(layer_input_size)
            layer_input_size = layer_pair.get_output_size_two_dimensional(layer_input_size)
            print("new layer_input_size: " + str(layer_input_size))
            print("number of output dimensions layer: " + str(result))
        return result

    def get_number_of_output_channels(self):
        last_layer_pair = self.multi_dimensional_lstm_layer_pairs[
            len(self.multi_dimensional_lstm_layer_pairs) - 1]
        return last_layer_pair.get_number_of_output_channels()

    def get_width_reduction_factor(self):
        result = 1
        for layer_pair_or_layer in self.multi_dimensional_lstm_layer_pairs:
            if isinstance(layer_pair_or_layer, MDLSTMLayerBlockStridedConvolutionLayerPair):
                result *= layer_pair_or_layer.block_strided_convolution.get_width_reduction_factor()
            elif isinstance(layer_pair_or_layer, BlockStridedConvolution):
                result *= layer_pair_or_layer.get_width_reduction_factor()

        return result

    def get_height_reduction_factor(self):
        result = 1
        for layer_pair_or_layer in self.multi_dimensional_lstm_layer_pairs:
            if isinstance(layer_pair_or_layer, MDLSTMLayerBlockStridedConvolutionLayerPair):
                result *= layer_pair_or_layer.block_strided_convolution.get_height_reduction_factor()
            elif isinstance(layer_pair_or_layer, BlockStridedConvolution):
                result *= layer_pair_or_layer.get_height_reduction_factor()

        return result

    def forward(self, x):
        network_input = x
        for layer_pair in self.multi_dimensional_lstm_layer_pairs:
            output = layer_pair(network_input)
            # print(">>> BlockMultiDimensionalLSTMLayerPairStacking.forward: - output.grad_fn "
            #      + str(output.grad_fn))
            # print("output.size(): " + str(output.size()))
            network_input = output
        return output
