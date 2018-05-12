from modules.multi_dimensional_lstm import MultiDimensionalLSTM
from util.tensor_chunking import TensorChunking
from modules.size_two_dimensional import SizeTwoDimensional
from torch.nn.modules.module import Module
import torch.nn as nn


# A MultiDimensionalBlockLSTM is a neural network layer that applies a MultiDimensionalLSTM
# to blocks of a 2 dimensional (4 dimensional including the batch and input channel dimensions)
# input tensor. The output is again a 4 dimensional output tensor, with the same width and
# height and batch size as the input, but possibly a different number of channels in the output
class MultiDimensionalBlockLSTM(Module):

    def __init__(self, multi_dimensional_lstm: MultiDimensionalLSTM,
                 tensor_chunking: TensorChunking):
        super(MultiDimensionalBlockLSTM, self).__init__()
        self.multi_dimensional_lstm = multi_dimensional_lstm
        self.tensor_chunking = tensor_chunking
        # self.state_convolutions = nn.ModuleList([])
        # self.state_convolutions.extend(self.multi_dimensional_lstm.state_convolutions)
        # self.state_convolutions.append(multi_dimensional_lstm)
        # print(">>> len(self.state_convolutions)" + str(len(self.state_convolutions)))


    @staticmethod
    def create_multi_dimensional_block_lstm(input_channels: int, hidden_states_size: int,
                                            batch_size:int,
                                            original_size: SizeTwoDimensional,
                                            block_size: SizeTwoDimensional,
                                            compute_multi_directional: bool,
                                            use_dropout: bool,
                                            nonlinearity="tanh"):
        tensor_chunking = TensorChunking.create_tensor_chunking(batch_size, original_size, block_size)
        blocks_parallel_batch_size = batch_size * tensor_chunking.number_of_feature_blocks_per_example

        multi_dimensional_lstm = MultiDimensionalLSTM.\
            create_multi_dimensional_lstm_fast(input_channels, hidden_states_size, blocks_parallel_batch_size,
                                               compute_multi_directional, use_dropout,
                                               nonlinearity)

        return MultiDimensionalBlockLSTM(multi_dimensional_lstm, tensor_chunking)

    def get_hidden_states_size(self):
        return self.multi_dimensional_lstm.get_hidden_states_size()

    def compute_multi_directional(self):
        return self.multi_dimensional_lstm.compute_multi_directional()

    def set_training(self, training):
        self.multi_dimensional_lstm.set_training(training)

    def forward(self, x):
        x_chunked = self.tensor_chunking.chunk_tensor_into_blocks_concatenate_along_batch_dimension(x)
        output = self.multi_dimensional_lstm(x_chunked)
        output_ordered_back_to_input_format = self.tensor_chunking.\
            dechunk_block_tensor_concatenated_along_batch_dimension(output)
        # print("output_ordered_back_to_input_format : " + str(output_ordered_back_to_input_format ))
        return output_ordered_back_to_input_format