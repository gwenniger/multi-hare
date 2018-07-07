import torch.nn as nn
import torch.nn.functional as F
from modules.multi_dimensional_rnn import StateUpdateBlock
from torch.nn.modules.module import Module
import torch
from modules.inside_model_gradient_clipping import InsideModelGradientClamping


# This class optimizes the computation of multiple 2d input convolutions
# that are computed from the same input, by computing them as a single
# convolution with more outputs, and then splitting the results.
# This class is very similar in spirit
# to ParallelMultipleStateWeightingsComputation, but is in fact somewhat
# simpler since:
# 1. It only has to compute one convolution per output element, not a
# convolution output pair as in the other class.
# 2. There is no need to shift outputs as in the other class.

class ParallelMultipleInputConvolutionsComputation(Module):
    def __init__(self, hidden_states_size: int,
                 number_of_input_convolutions: int,
                 output_states_size: int,
                 parallel_convolution: nn.Conv2d,
                 clamp_gradients: bool,
                 use_dropout: bool,
                 training: bool):
        super(ParallelMultipleInputConvolutionsComputation, self).__init__()
        self.hidden_states_size = hidden_states_size
        self.number_of_input_convolutions = number_of_input_convolutions
        self.output_states_size = output_states_size
        self.parallel_convolution = parallel_convolution
        self.clamp_gradients = clamp_gradients
        self.use_dropout = use_dropout
        self.training = training

    @staticmethod
    def create_parallel_multiple_input_convolutions_computation(input_channels: int,
                                                                hidden_states_size: int,
                                                                number_of_input_convolutions: int,
                                                                clamp_gradients: bool,
                                                                use_dropout: bool):
        output_states_size = hidden_states_size * number_of_input_convolutions

        # parallel_convolution = nn.Conv1d(hidden_states_size, output_states_size, 1)

        parallel_convolution = nn.Conv2d(input_channels, output_states_size, 1)

        # Xavier weight initialization
        torch.nn.init.xavier_uniform_(parallel_convolution.weight)

        return ParallelMultipleInputConvolutionsComputation(hidden_states_size, number_of_input_convolutions,
                                                            output_states_size, parallel_convolution, clamp_gradients,
                                                            use_dropout,
                                                            True)

    # How to do dropout in pytorch:
    # https://discuss.pytorch.org/t/dropout-functional-api-advantages-disadvantages/181/4
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    # Where to apply dropout:
    # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
    def compute_convolution_result(self, input_tensor):
        if self.use_dropout:
                # print("Applying dropout...")
                # TODO: which probability to use for dropout?
                result = F.dropout(self.parallel_convolution(input_tensor), p=0.2, training=self.training)
                return result
        result = self.parallel_convolution(input_tensor)

        if self.clamp_gradients:
            # print("ParallelMultipleStateWeightingsComputation - register gradient clamping...")
            # Create a 1d convolution with clamping of the gradient
            result = InsideModelGradientClamping.register_gradient_clamping_default_clamping_bound(result)

        return result

    def get_result_range_start_index(self, result_element_index):
        return self.hidden_states_size * result_element_index

    def get_result_range_end_index(self, result_element_index):
        return self.hidden_states_size * (result_element_index + 1)

    def compute_result_and_split_into_output_elements(self, input_tensor):
        result = list([])

        convolution_result = self.compute_convolution_result(input_tensor)
        # print(">>> compute_result_and_split_into_output_elements - convolution_result.size(): " +
        #      str(convolution_result.size()))

        for i in range(0, self.number_of_input_convolutions):
            range_begin = self.get_result_range_start_index(i)
            range_end = self.get_result_range_end_index(i)
            # print("range begin: " + str(range_begin) + " range end: " + str(range_end))
            element = convolution_result[:, range_begin:range_end, :]

            result.append(element)
        return result

    def get_state_convolutions_as_list(self):
        return list([self.parallel_convolution])

    # When testing the model, training should be set to false
    def set_training(self, training):
        self.training = training

    # This class extends Module so as to make sure that the parameters
    # are properly copied (to the right cuda device) when using nn.DataParallel(model)
    # and the to(device) method from  the Module base class
    # http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html
    # The class is not however meant to be used as a stand-alone Module, so forward
    # is not implemented
    def forward(self, x):
        raise NotImplementedError
