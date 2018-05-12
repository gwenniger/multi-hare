from modules.block_multi_dimensional_lstm import BlockMultiDimensionalLSTM
from modules.block_strided_convolution import BlockStridedConvolution
from torch.nn.modules.module import Module
from modules.size_two_dimensional import SizeTwoDimensional
from util.tensor_chunking import TensorChunking

class BlockMultiDimensionalLSTMLayerPair(Module):

    def __init__(self, block_multi_dimensional_lstm: BlockMultiDimensionalLSTM,
                 block_strided_convolution: BlockStridedConvolution):
        super(BlockMultiDimensionalLSTMLayerPair, self).__init__()
        self.block_multi_dimensional_lstm = block_multi_dimensional_lstm
        self.block_strided_convolution =  block_strided_convolution

    @staticmethod
    def create_block_multi_dimensional_lstm_layer_pair(
            input_channels: int, mdlstm_hidden_states_size: int,
            output_channels: int, block_size: SizeTwoDimensional,
            compute_multi_directional: bool, use_dropout: bool, nonlinearity="tanh"):
        block_multi_dimensional_lstm = \
            BlockMultiDimensionalLSTM.create_block_multi_dimensional_lstm(
                input_channels, mdlstm_hidden_states_size, block_size, compute_multi_directional, use_dropout,
                nonlinearity)

        block_strided_convolution = BlockStridedConvolution.\
            create_block_strided_convolution(mdlstm_hidden_states_size, output_channels,
                                             block_size, nonlinearity)

        return BlockMultiDimensionalLSTMLayerPair(block_multi_dimensional_lstm, block_strided_convolution)

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        block_size = self.block_multi_dimensional_lstm.block_size
        tensor_chunking = TensorChunking.create_tensor_chunking(input_size, block_size)
        feature_blocks_per_example = tensor_chunking.number_of_feature_blocks_per_example
        result = feature_blocks_per_example * self.block_strided_convolution.output_channels
        return result

    def set_training(self, training):
        self.block_multi_dimensional_lstm.set_training(training)

    def forward(self, x):
        block_mdlstm_output = self.block_multi_dimensional_lstm(x)
        convolution_output = self.block_strided_convolution(block_mdlstm_output)
        return convolution_output
