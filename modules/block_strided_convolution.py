from modules.size_two_dimensional import SizeTwoDimensional
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_chunking import TensorChunking


class BlockStridedConvolution(Module):

    def __init__(self, input_channels: int, output_channels: int, block_size: SizeTwoDimensional,
                 nonlinearity="tanh"):
        super(BlockStridedConvolution, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_size = block_size
        self.nonlinearity = nonlinearity
        # What types of convolutions are there:
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        # https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
        # Don't use bias in the convolution layer (?). This is suggested by
        # "Dropout improves Recurrent Neural Networks for Handwriting Recognition"
        self.convolution = nn.Conv2d(self.input_channels, self.output_channels,
                                     (block_size.height, block_size.width),
                                     stride=(block_size.height, block_size.width),
                                     bias=True)

    @staticmethod
    def create_block_strided_convolution(input_channels: int, output_channels: int, block_size: SizeTwoDimensional,
                 nonlinearity="tanh"):
        return BlockStridedConvolution(input_channels, output_channels, block_size, nonlinearity)

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
        convolution_output = self.convolution(x)
        # print("convolution output: " + str(convolution_output))
        result = self.get_activation_function()(convolution_output)
        return result
