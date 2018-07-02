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
            output_channels: int, mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            compute_multi_directional: bool, clamp_gradients: bool,
            use_dropout: bool,
            inputs_and_outputs_are_lists: bool,
            nonlinearity="tanh"):
        block_multi_dimensional_lstm = \
            BlockMultiDimensionalLSTM.create_block_multi_dimensional_lstm(
                input_channels, mdlstm_hidden_states_size, mdlstm_block_size, compute_multi_directional,
                clamp_gradients, use_dropout,
                inputs_and_outputs_are_lists,
                nonlinearity)

        block_strided_convolution = BlockStridedConvolution.\
            create_block_strided_convolution(mdlstm_hidden_states_size, output_channels,
                                             block_strided_convolution_block_size,
                                             clamp_gradients, inputs_and_outputs_are_lists, nonlinearity)

        return BlockMultiDimensionalLSTMLayerPair(block_multi_dimensional_lstm, block_strided_convolution)

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        return self.block_strided_convolution.get_number_of_output_dimensions(input_size)

    def get_output_size_two_dimensional(self, input_size: SizeTwoDimensional):
        return self.block_strided_convolution.get_output_size_two_dimensional(input_size)

    def get_number_of_output_channels(self):
        return self.block_strided_convolution.output_channels

    def set_training(self, training):
        self.block_multi_dimensional_lstm.set_training(training)

    def forward(self, x):
        block_mdlstm_output = self.block_multi_dimensional_lstm(x)
        convolution_output = self.block_strided_convolution(block_mdlstm_output)
        # print("convolution_output.size():" + str( convolution_output.size()))
        return convolution_output
