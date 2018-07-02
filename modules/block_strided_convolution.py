from modules.size_two_dimensional import SizeTwoDimensional
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_chunking import TensorChunking
from util.tensor_list_chunking import TensorListChunking
from modules.inside_model_gradient_clipping import InsideModelGradientClamping


class BlockStridedConvolution(Module):

    def __init__(self, input_channels: int, output_channels: int, block_size: SizeTwoDimensional,
                 clamp_gradients: bool,
                 input_and_output_are_lists: bool,
                 nonlinearity="tanh"):
        super(BlockStridedConvolution, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_size = block_size
        self.nonlinearity = nonlinearity
        self.clamp_gradients = clamp_gradients
        self.input_and_output_are_list = input_and_output_are_lists
        # What types of convolutions are there:
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        # https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
        # Don't use bias in the convolution layer (?). This is suggested by
        # "Dropout improves Recurrent Neural Networks for Handwriting Recognition"
        self.convolution = nn.Conv2d(self.input_channels, self.output_channels,
                                     (block_size.height, block_size.width),
                                     stride=(block_size.height, block_size.width),
                                     bias=True)

        # Initialize the convolution with the
        # Xavier Glorot scheme
        nn.init.xavier_uniform_(self.convolution.weight)

        print("BlockStridedConvolution - clamp_gradients: " + str(clamp_gradients))

    @staticmethod
    def create_block_strided_convolution(input_channels: int, output_channels: int, block_size: SizeTwoDimensional,
                                         clamp_gradients: bool,inputs_and_outputs_are_lists: bool,
                 nonlinearity="tanh"):
        return BlockStridedConvolution(input_channels, output_channels, block_size,
                                       clamp_gradients, inputs_and_outputs_are_lists, nonlinearity)

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
        print("feature_blocks_per_example : " + str(feature_blocks_per_example))
        result = feature_blocks_per_example * self.output_channels
        return result

    def get_output_size_two_dimensional(self, input_size: SizeTwoDimensional):
        block_size = self.block_size
        height = int(input_size.height / block_size.height)
        width = int(input_size.width / block_size.width)
        return SizeTwoDimensional.create_size_two_dimensional(height, width)

    def set_training(self, training):
        return

    def forward(self, x):
        if self.input_and_output_are_list:
            tensor_list_chunking = TensorListChunking.create_tensor_list_chunking(x, self.block_size)
            x_chunked = tensor_list_chunking.chunk_tensor_list_into_blocks_concatenate_along_batch_dimension(x)
        else:
            x_chunked = x

        convolution_output = self.convolution(x_chunked)
        if self.clamp_gradients:
            convolution_output = InsideModelGradientClamping.register_gradient_clamping(convolution_output)
        # print("convolution output: " + str(convolution_output))
        result = self.get_activation_function()(convolution_output)

        # Tanh and sigmoid have a derivative that is not higher than 1,
        # so they should not give large gradients in the backward pass
        # if self.clamp_gradients:
            # print("BlockStridedConvolution.forward - clamp gradients")
        #result = InsideModelGradientClamping.register_gradient_clamping(result)

        # If the input and output are lists, the output of the convolution
        # and activation function must again be converted back to the original list
        # format
        if self.input_and_output_are_list:
            convolution_output_size = SizeTwoDimensional.create_size_two_dimensional(1, 1)
            output_ordered_back_to_input_format = tensor_list_chunking. \
                dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size(result,
                                                                                           convolution_output_size)
            return output_ordered_back_to_input_format

        return result

    def get_width_reduction_factor(self):
        return self.block_size.width

    def get_height_reduction_factor(self):
        return self.block_size.height

