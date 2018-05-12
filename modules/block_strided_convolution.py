from modules.size_two_dimensional import SizeTwoDimensional
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


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
        self.convolution = nn.Conv2d(self.input_channels, self.output_channels,
                                     (block_size.height, block_size.width),
                                     stride=(block_size.height, block_size.width))

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

    def forward(self, x):
        convolution_output = self.convolution(x)
        # print("convolution output: " + str(convolution_output))
        result = self.get_activation_function()(convolution_output)
        return result
