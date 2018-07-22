from modules.block_multi_dimensional_lstm import BlockMultiDimensionalLSTM
from modules.block_strided_convolution import BlockStridedConvolution
from modules.multi_dimensional_lstm import MultiDimensionalLSTM
from torch.nn.modules.module import Module
from modules.size_two_dimensional import SizeTwoDimensional
from util.tensor_chunking import TensorChunking


class MDLSTMLayerBlockStridedConvolutionLayerPair(Module):

    def __init__(self, mdlstm_layer: Module,
                 block_strided_convolution: BlockStridedConvolution):
        super(MDLSTMLayerBlockStridedConvolutionLayerPair, self).__init__()
        self.mdlstm_layer = mdlstm_layer
        self.block_strided_convolution = block_strided_convolution

    @staticmethod
    def create_block_mdlstm_block_strided_convolution_layer_pair(
            input_channels: int, mdlstm_hidden_states_size: int,
            output_channels: int, mdlstm_block_size: SizeTwoDimensional,
            block_strided_convolution_block_size: SizeTwoDimensional,
            compute_multi_directional: bool, clamp_gradients: bool,
            use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            nonlinearity="tanh"):

        print("Create {block-mdlstm, block-strided_convolution} layer_pair...")

        block_multi_dimensional_lstm = \
            BlockMultiDimensionalLSTM.create_block_multi_dimensional_lstm(
                input_channels, mdlstm_hidden_states_size, mdlstm_block_size, compute_multi_directional,
                clamp_gradients, use_dropout,
                nonlinearity)

        block_strided_convolution = BlockStridedConvolution.\
            create_block_strided_convolution(mdlstm_hidden_states_size, output_channels,
                                             block_strided_convolution_block_size,
                                             clamp_gradients,
                                             use_bias_with_block_strided_convolution,
                                             nonlinearity)

        return MDLSTMLayerBlockStridedConvolutionLayerPair(block_multi_dimensional_lstm, block_strided_convolution)

    @staticmethod
    def create_mdlstm_block_strided_convolution_layer_pair(
            layer_index,
            input_channels: int, mdlstm_hidden_states_size: int,
            output_channels: int,
            block_strided_convolution_block_size: SizeTwoDimensional,
            compute_multi_directional: bool, clamp_gradients: bool,
            use_dropout: bool,
            use_bias_with_block_strided_convolution: bool,
            use_example_packing: bool,
            nonlinearity="tanh"):

        print("Create {mdlstm,_block-strided_convolution} layer_pair...")
        multi_dimensional_lstm = MultiDimensionalLSTM.\
            create_multi_dimensional_lstm_fast(layer_index, input_channels, mdlstm_hidden_states_size,
                                               compute_multi_directional,
                                               clamp_gradients,
                                               use_dropout,
                                               use_example_packing,
                                               nonlinearity)
        block_strided_convolution = BlockStridedConvolution. \
            create_block_strided_convolution(mdlstm_hidden_states_size, output_channels,
                                             block_strided_convolution_block_size,
                                             clamp_gradients,
                                             use_bias_with_block_strided_convolution,
                                             use_example_packing,
                                             nonlinearity)

        return MDLSTMLayerBlockStridedConvolutionLayerPair(multi_dimensional_lstm, block_strided_convolution)

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        return self.block_strided_convolution.get_number_of_output_dimensions(input_size)

    def get_output_size_two_dimensional(self, input_size: SizeTwoDimensional):
        return self.block_strided_convolution.get_output_size_two_dimensional(input_size)

    def get_number_of_output_channels(self):
        return self.block_strided_convolution.output_channels

    def set_training(self, training):
        self.mdlstm_layer.set_training(training)

    def forward(self, x):
        mdlstm_layer_output = self.mdlstm_layer(x)
        convolution_output = self.block_strided_convolution(mdlstm_layer_output)
        # print("convolution_output.size():" + str( convolution_output.size()))
        return convolution_output
