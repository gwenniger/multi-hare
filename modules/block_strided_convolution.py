from modules.size_two_dimensional import SizeTwoDimensional
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_chunking import TensorChunking
from util.tensor_list_chunking import TensorListChunking
from modules.inside_model_gradient_clipping import InsideModelGradientClamping
from util.tensor_utils import TensorUtils
from modules.module_io_structuring import ModuleIOStructuring
import torch
import util.timing


class BlockStridedConvolution(Module):

    def __init__(self, layer_pair_index: int,
                 input_channels: int, output_channels: int, block_size: SizeTwoDimensional,
                 clamp_gradients: bool,
                 use_bias: bool,
                 use_example_packing: bool,
                 comput_multi_directional: bool,
                 share_weights_across_directions: bool,
                 convolution,
                 nonlinearity="tanh"):
        super(BlockStridedConvolution, self).__init__()
        self.layer_pair_index = layer_pair_index,
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_size = block_size
        self.nonlinearity = nonlinearity
        self.clamp_gradients = clamp_gradients
        self.use_example_packing = use_example_packing
        self.compute_multi_directional = comput_multi_directional
        self.share_weights_across_directions = share_weights_across_directions
        # What types of convolutions are there:
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        # https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

        self.convolution = convolution

        if use_bias:
            print("WARNING: using bias with block_strided_Convolution")
        else:
            print("Creating block_strided_convolution without bias parameters...")

        print("BlockStridedConvolution - clamp_gradients: " + str(clamp_gradients))

    @staticmethod
    def create_convolution(input_channels: int,
                           output_channels: int,
                           block_size,
                           use_bias: bool,
                           number_of_directions: int):
        # Don't use bias in the convolution layer (?). This is suggested by
        # "Dropout improves Recurrent Neural Networks for Handwriting Recognition"

        if number_of_directions > 1:
            convolution = nn.Conv2d(input_channels * number_of_directions,
                                    output_channels * number_of_directions,
                                    (block_size.height, block_size.width),
                                    stride=(block_size.height, block_size.width),
                                    bias=use_bias, groups=number_of_directions)
        else:
            convolution = nn.Conv2d(input_channels, output_channels,
                                    (block_size.height, block_size.width),
                                    stride=(block_size.height, block_size.width),
                                    bias=use_bias)

        # Initialize the convolution with the
        # Xavier Glorot scheme
        nn.init.xavier_uniform_(convolution.weight)
        return convolution

    @staticmethod
    def create_block_strided_convolution(layer_pair_index: int, input_channels: int, output_channels: int, block_size: SizeTwoDimensional,
                                         clamp_gradients: bool, use_bias: bool, use_example_packing: bool,
                                         compute_multi_directional: bool,
                                         share_weights_across_directions: bool,
                                         nonlinearity="tanh"):

        print("Creating block-strided convolution for layer pair: " + str(layer_pair_index) +
              " sharing weights across directions:  " + str(share_weights_across_directions))

        if compute_multi_directional and not share_weights_across_directions:
            number_of_directions = 4
        else:
            number_of_directions = 1
        convolution = BlockStridedConvolution.create_convolution(input_channels, output_channels,
                                                                 block_size, use_bias,
                                                                 number_of_directions)

        return BlockStridedConvolution(layer_pair_index, input_channels, output_channels, block_size,
                                       clamp_gradients, use_bias, use_example_packing,
                                       compute_multi_directional,
                                       share_weights_across_directions,
                                       convolution, nonlinearity)

    def get_activation_function(self):
        if self.nonlinearity == "tanh":
            activation_function = F.tanh
        elif self.nonlinearity == "relu":
            # func = self._backend.RNNReLUCell
            activation_function = F.relu
        elif self.nonlinearity == "sigmoid":
            activation_function = F.sigmoid
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return activation_function

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        block_size = self.block_size
        tensor_chunking = TensorChunking.create_tensor_chunking(input_size, block_size)
        feature_blocks_per_example = tensor_chunking.number_of_feature_blocks_per_example
        # print("block_strided_convolution - feature_blocks_per_example : " + str(feature_blocks_per_example))
        result = feature_blocks_per_example * self.output_channels
        return result

    def get_output_size_two_dimensional(self, input_size: SizeTwoDimensional):
        block_size = self.block_size
        height = int(input_size.height / block_size.height)
        width = int(input_size.width / block_size.width)
        return SizeTwoDimensional.create_size_two_dimensional(height, width)

    def set_training(self, training):
        return

    def compute_forward_one_directional(self, x):
        convolution_output = self.convolution(x)
        # TensorUtils.print_max(convolution_output, "block_strided_convolution - convolution_output")
        if self.clamp_gradients:
            # convolution_output = InsideModelGradientClamping.register_gradient_clamping_default_clamping_bound(
            #     convolution_output, "block_strided_convolution - convolution_output")
            convolution_output = InsideModelGradientClamping.register_gradient_clamping(
                convolution_output, 10, True, "block_strided_convolution - convolution_output")
        # print("convolution output: " + str(convolution_output))
        result = self.get_activation_function()(convolution_output)
        # TensorUtils.print_max(result, "block_strided_convolution - result")

        # Tanh and sigmoid have a derivative that is not higher than 1,
        # so they should not give large gradients in the backward pass
        # if self.clamp_gradients:
        # print("BlockStridedConvolution.forward - clamp gradients")
        # result = InsideModelGradientClamping.register_gradient_clamping(result)
        return result

    @staticmethod
    def chunk_four_parts_on_channel_dimension_and_sum(tensor):
        results_per_direction = torch.chunk(tensor, 4, 1)
        result = torch.sum(torch.stack(results_per_direction, 0), 0)
        return result

    def compute_forward_from_chunked_input(self, x_chunked, tensor_list_chunking):

        result = self.compute_forward_one_directional(x_chunked)

        # If the input and output are lists, the output of the convolution
        # and activation function must again be converted back to the original list
        # format
        # if self.input_and_output_are_list:
        #     convolution_output_size = SizeTwoDimensional.create_size_two_dimensional(1, 1)
        #     output_ordered_back_to_input_format = tensor_list_chunking. \
        #         dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size(result,
        #                                                                                    convolution_output_size)
        #     return output_ordered_back_to_input_format

        # print("block_strided_convolution - result.size(): " + str(result.size()))

        # Sum the results for multiple directions contained in chunks of the result
        # If the weights are shared across directions, this summation has already been done over the inputs
        # before computing the convolution
        if self.compute_multi_directional and not self.share_weights_across_directions:
            result = BlockStridedConvolution.chunk_four_parts_on_channel_dimension_and_sum(result)
            # result = TensorUtils.sum_list_of_tensors(results_per_direction)

        if self.use_example_packing:
            # print("block_strided_convolution - use example packing")
            result = tensor_list_chunking. \
                dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size(result,
                                                                                           SizeTwoDimensional(1, 1))
        return result

    """
    This implementation splits the chunk tensor into portions that are processed consecutively,
    to exploit the fact that in the summation the convolution results for the different 
    directions are collapsed. This would be expected to save memory, since only the summed
    sub-tensors need to be saved till the end of the loop. Whether this also plays out 
    given the dependencies for the back-propagation has to be seen  
    """
    def compute_forward_from_chunked_input_using_portions(self, x_chunked, tensor_list_chunking):

        # Sum the results for multiple directions contained in chunks of the result
        if self.compute_multi_directional:

            # print("compute_forward_from_chunked_input_using_portions - x_chunked.size(): " +
            #       str(x_chunked.size()))

            cat_list = list([])
            data_portions = torch.chunk(x_chunked, 4, 0)
            for data_portion in data_portions:
                data_portion_conv_result = self.compute_forward_one_directional(data_portion)
                data_portion_results_per_direction = torch.chunk(data_portion_conv_result, 4, 1)
                data_portion_result = torch.sum(torch.stack(data_portion_results_per_direction, 0), 0)
                cat_list.append(data_portion_result)
            result = torch.cat(cat_list, 0)
        else:
            result = self.compute_forward_one_directional(x_chunked)

        if self.use_example_packing:
            # print("block_strided_convolution - use example packing")
            result = tensor_list_chunking. \
                dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size(result,
                                                                                           SizeTwoDimensional(1, 1))
        return result

    def compute_x_chunked_and_tensor_list_chunking_list(self, x):
        # Tensor list chunking expects a list of 3-D tensors as input, but x
        # obtained from MDLSTM is a list of 4-D tensors, so must convert
        x_three_dim = list([])
        for tensor in x:
            # print("block_strided_convolution.forward - tensor.size(): " + str(tensor.size()))
            x_three_dim.append(tensor.squeeze(0))

        # FIXME: For 4-directional MDLSTM, tensor list chunking must be done for 4 different
        # directions.
        # Then convolutions must be computed for every dimension separately and summed
        # This is achieved efficiently using a single convolution using groups.

        tensor_list_chunking = TensorListChunking.create_tensor_list_chunking(x_three_dim, self.block_size)
        x_chunked = \
            tensor_list_chunking.chunk_tensor_list_into_blocks_concatenate_along_batch_dimension(x_three_dim, False)

        # # Debugging: check that the de-chunked version recovers the original
        # ModuleIOStructuring. \
        #     check_dechunking_chunked_tensor_list_recovers_original(tensor_list_chunking, x_three_dim, x_chunked)
        return x_chunked, tensor_list_chunking

    @staticmethod
    def sum_channels_four_directions(x_chunked):
        # print("x_chunked.size(): " + str(x_chunked.size()))
        x_chunked_split_by_channels = torch.chunk(x_chunked, 4, 1)
        stacked_channels = torch.stack(x_chunked_split_by_channels, 0)
        # print("stacked_channels.size(): " + str(stacked_channels.size()))
        x_chunked_summed_channels = torch.sum(stacked_channels, 0)
        # print("x_chunked_summed_channels.size(): " + str(x_chunked_summed_channels.size()))
        return x_chunked_summed_channels

    def forward(self, x):

        """
        :param x: (If Using examples-packing) A list of example tensors, each tensor
        of dimension [1, InputChannels, Height, Width]

        :return:
        (If Using examples-packing) Returns output in the form of a list, one
        list entry for each example, with entries of the form
        [OutputChannels, Height, Width]
        So for example an
        input image of size 4 * 24, using 10 output channels will yield output list
        elements of size [10, 4, 24]
        """

        # time_start_network_forward = util.timing.date_time_start()

        # print("Block-strided convolution input sizes: ")
        # for element in x:
        #     print("element.size()" + str(element.size()))

        if self.use_example_packing:
            x_chunked, tensor_list_chunking = \
                self.compute_x_chunked_and_tensor_list_chunking_list(x)

            if self.compute_multi_directional:

                if self.share_weights_across_directions:
                    x_chunked_summed_channels = BlockStridedConvolution.sum_channels_four_directions(x_chunked)
                    activations_summed = self.compute_forward_from_chunked_input(
                        x_chunked_summed_channels, tensor_list_chunking)

                else:
                    # activations_summed = self.compute_forward_from_chunked_input(
                    #     x_chunked, tensor_list_chunking)
                    activations_summed = self.compute_forward_from_chunked_input_using_portions(
                        x_chunked, tensor_list_chunking)
                    # print("block_strided_convolution - activations[0].size(): " + str(activations[0].size()))
                    # Chunk the list of activations per tensor into a list of activations per tensor
                    # for each direction
                    #list_of_activations_lists = TensorUtils.chunk_list_of_tensors_along_dimension(activations, 4, 0)
                    #activations_summed = TensorUtils.sum_lists_of_tensor_lists_element_wise(list_of_activations_lists)
                    # print("block_strided_convolution - activations_summed[0].size(): "
                    # + str(activations_summed[0].size()))

                    # print("Block-strided convolution - Time used for network forward: "
                    #       + str(util.timing.milliseconds_since(time_start_network_forward)))

                    # for i, element in enumerate(activations_summed):
                    #     print("Block-strided convolution - result[" + str(i) + "].size(): " + str(element.size()))
                    #

                # print("Block-strided convolution output sizes: ")
                # for element in activations_summed:
                #     print("element.size()" + str(element.size()))

                return activations_summed
            else:
                activations = self.compute_forward_from_chunked_input(
                    x_chunked, tensor_list_chunking)
                return activations
        else:
            if self.compute_multi_directional:
                if self.share_weights_across_directions:
                    x_summed_channels = BlockStridedConvolution.sum_channels_four_directions(x)
                    return self.compute_forward_one_directional(x_summed_channels)
                else:
                    result = self.compute_forward_one_directional(x)
                    result = BlockStridedConvolution.chunk_four_parts_on_channel_dimension_and_sum(result)
                    return result
            else:
                return self.compute_forward_one_directional(x)

    def get_width_reduction_factor(self):
        return self.block_size.width

    def get_height_reduction_factor(self):
        return self.block_size.height

