from modules.multi_dimensional_lstm import MultiDimensionalLSTM
from util.tensor_chunking import TensorChunking
from modules.size_two_dimensional import SizeTwoDimensional
from torch.nn.modules.module import Module


# A MultiDimensionalBlockLSTM is a neural network layer that applies a MultiDimensionalLSTM
# to blocks of a 2 dimensional (4 dimensional including the batch and input channel dimensions)
# input tensor. The output is again a 4 dimensional output tensor, with the same width and
# height and batch size as the input, but possibly a different number of channels in the output
class BlockMultiDimensionalLSTM(Module):

    def __init__(self, multi_dimensional_lstm: MultiDimensionalLSTM, block_size: SizeTwoDimensional):
        super(BlockMultiDimensionalLSTM, self).__init__()
        self.multi_dimensional_lstm = multi_dimensional_lstm
        self.block_size = block_size

    @staticmethod
    def create_block_multi_dimensional_lstm(input_channels: int, hidden_states_size: int,
                                            block_size: SizeTwoDimensional,
                                            compute_multi_directional: bool,
                                            use_dropout: bool,
                                            nonlinearity="tanh"):
        multi_dimensional_lstm = MultiDimensionalLSTM.\
            create_multi_dimensional_lstm_fast(input_channels, hidden_states_size,
                                               compute_multi_directional, use_dropout,
                                               nonlinearity)

        return BlockMultiDimensionalLSTM(multi_dimensional_lstm, block_size)

    def get_hidden_states_size(self):
        return self.multi_dimensional_lstm.get_hidden_states_size()

    def compute_multi_directional(self):
        return self.multi_dimensional_lstm.compute_multi_directional()

    def set_training(self, training):
        self.multi_dimensional_lstm.set_training(training)

    def get_number_of_output_dimensions(self, input_size: SizeTwoDimensional):
        result = input_size.height * input_size.width \
                 * self.get_hidden_states_size()
        if self.compute_multi_directional():
            result = result * 4
        return result

    def get_output_size_two_dimensional(self, input_size: SizeTwoDimensional):
        return input_size

    def forward(self, x):
        original_size = SizeTwoDimensional.create_size_two_dimensional(x.size(2), x.size(3))
        # Tensor chunking is created dynamically, so that every batch may have a different
        # two-dimensional size (within each batch, examples must still be of the same size)
        # print("BlockMultiDimensionalLSTM - self.block_size: " + str(self.block_size))
        tensor_chunking = TensorChunking.create_tensor_chunking(original_size, self.block_size)

        x_chunked = tensor_chunking.chunk_tensor_into_blocks_concatenate_along_batch_dimension(x)
        output = self.multi_dimensional_lstm(x_chunked)
        output_ordered_back_to_input_format = tensor_chunking.\
            dechunk_block_tensor_concatenated_along_batch_dimension(output)
        # print("output_ordered_back_to_input_format : " + str(output_ordered_back_to_input_format ))
        return output_ordered_back_to_input_format