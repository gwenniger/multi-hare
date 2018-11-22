import torch.nn as nn
import torch.nn.functional as F
from modules.multi_dimensional_rnn import StateUpdateBlock
from torch.nn.modules.module import Module
import torch
from modules.inside_model_gradient_clipping import InsideModelGradientClamping
from modules.gradient_clamped_module import GradientClampedModule
from modules.xavier_weight_initialization_correction_for_grouping import XavierWeightInitializationCorrectionForGrouping

# This class optimizes the computation of multiple 2d input convolutions
# that are computed from the same input, by computing them as a single
# convolution with more outputs, and then splitting the results.
# This class is very similar in spirit
# to ParallelMultipleStateWeightingsComputation, but is in fact somewhat
# simpler since:
# 1. It only has to compute one convolution per output element, not a
# convolution output pair as in the other class.
# 2. There is no need to shift outputs as in the other class.

# The parameter number_of_groups can be used to further parallelize
# the computation of multiple groups of 2d convolutions. Choosing the
# number of groups > 1, each group has its own set of convolutions
# that processes one chunk of the input and produces one chunk of the output

class ParallelMultipleInputConvolutionsComputation(Module):
    def __init__(self, hidden_states_size: int,
                 number_of_input_convolutions: int,
                 output_states_size: int,
                 parallel_convolution: nn.Conv2d,
                 clamp_gradients: bool,
                 use_dropout: bool,
                 training: bool,
                 number_of_groups: int):
        super(ParallelMultipleInputConvolutionsComputation, self).__init__()
        self.hidden_states_size = hidden_states_size
        self.number_of_input_convolutions = number_of_input_convolutions
        self.output_states_size = output_states_size
        self.parallel_convolution = parallel_convolution
        self.clamp_gradients = clamp_gradients
        self.use_dropout = use_dropout
        self.training = training
        self.number_of_groups = number_of_groups

        print("ParallelMultipleInputConvolutions - clamp_gradients: " + str(clamp_gradients))

    @staticmethod
    def create_parallel_multiple_input_convolutions_computation(input_channels_per_group: int,
                                                                hidden_states_size: int,
                                                                number_of_input_convolutions: int,
                                                                clamp_gradients: bool,
                                                                use_dropout: bool,
                                                                number_of_groups=1):

        # Compute the required output states size, which is a function of the hidden states size,
        # the number of input convolutions to be computed in parallel, and the number of groups
        # to be computed in parallel
        output_states_size = hidden_states_size * number_of_input_convolutions * number_of_groups
        # Compute the required input channels size, which is a function of the input channels per
        # group and the number of groups
        input_channels = input_channels_per_group * number_of_groups

        # parallel_convolution = nn.Conv1d(hidden_states_size, output_states_size, 1)

        parallel_convolution = nn.Conv2d(input_channels, output_states_size, 1, groups=number_of_groups)

        if clamp_gradients:
            parallel_convolution = GradientClampedModule(parallel_convolution)


        # Xavier weight initialization
        print("Parallel_multiple_input_convolution_computation - Xavier weight initialization")
        torch.nn.init.xavier_uniform_(parallel_convolution.weight)
        # Compensate the weights for the fact that there are multiple groups: effectively there is a number
        # of virtual independent layers, equal to the number of groups. Therefore the weights should be
        # re-scaled by a factor sqrt(groups) to get the right initialization for each of the separate virtual
        # layers
        XavierWeightInitializationCorrectionForGrouping. \
            correct_xavier_uniform_initialized_weights_for_grouping(parallel_convolution.weight, number_of_groups)

        return ParallelMultipleInputConvolutionsComputation(hidden_states_size, number_of_input_convolutions,
                                                            output_states_size, parallel_convolution, clamp_gradients,
                                                            use_dropout,
                                                            True, number_of_groups)

    # How to do dropout in pytorch:
    # https://discuss.pytorch.org/t/dropout-functional-api-advantages-disadvantages/181/4
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    # Where to apply dropout:
    # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
    def compute_convolution_result(self, input_tensor):
        # print("parallel_multiple_input_convolutions_computation - compute_convolution_result")
        # print(" - input_tensor.size(): " + str(input_tensor.size()))

        if self.use_dropout:
                # print("Applying dropout...")
                # TODO: which probability to use for dropout?
                result = F.dropout(self.parallel_convolution(input_tensor), p=0.2, training=self.training)
                return result
        result = self.parallel_convolution(input_tensor)

        if self.clamp_gradients:
            # print("ParallelMultipleStateWeightingsComputation - register gradient clamping...")
            # Create a 1d convolution with clamping of the gradient
            result = InsideModelGradientClamping.\
                register_gradient_clamping_default_clamping_bound(result, "parallel_multiple_inputs_convolution")

        return result

    def get_result_range_start_index(self, result_element_index):
        if self.number_of_groups != 1:
            raise RuntimeError("Error: this function should not be used unless number_of_groups = 1")
        return self.get_group_result_range_start_index(result_element_index, 0)

    def get_result_range_end_index(self, result_element_index):
        if self.number_of_groups != 1:
            raise RuntimeError("Error: this function should not be used unless number_of_groups = 1")

        return self.get_group_result_range_end_index(result_element_index, 0)

    def get_output_states_per_group(self):
        return self.hidden_states_size * self.number_of_input_convolutions

    def get_group_result_range_start_index(self, result_element_index, group_index):
        group_offset = group_index * self.get_output_states_per_group()
        return group_offset + self.hidden_states_size * result_element_index

    def get_group_result_range_end_index(self, result_element_index, group_index):
        group_offset = group_index * self.get_output_states_per_group()
        return group_offset + self.hidden_states_size * (result_element_index + 1)

    def split_result_into_output_elements_for_single_group_computation(self, convolution_result):
        # result = list([])
        # for i in range(0, self.number_of_input_convolutions):
        #     range_begin = self.get_result_range_start_index(i)
        #     range_end = self.get_result_range_end_index(i)
        #     # print("range begin: " + str(range_begin) + " range end: " + str(range_end))
        #     element = convolution_result[:, range_begin:range_end, :]
        #
        #     result.append(element)
        # return result

        return torch.chunk(convolution_result, self.number_of_input_convolutions, 1)

    def split_result_into_output_elements_for_multiple_group_computation(self, convolution_result):
        result = list([])

        convolution_results_split_by_group = torch.chunk(convolution_result, self.number_of_groups,
                                                         1)
        for convolution_result_group in convolution_results_split_by_group:
            convolution_results_split_by_group_and_index = \
                torch.chunk(convolution_result_group, self.number_of_input_convolutions, 1)
            result.append(convolution_results_split_by_group_and_index)

        # for group_index in range(0, self.number_of_groups):
        #
        #     group_results_list = list([])
        #     for i in range(0, self.number_of_input_convolutions):
        #         range_begin = self.get_group_result_range_start_index(i, group_index)
        #         range_end = self.get_group_result_range_end_index(i, group_index)
        #         # print("range begin: " + str(range_begin) + " range end: " + str(range_end))
        #         element = convolution_result[:, range_begin:range_end, :]
        #         group_results_list.append(element)
        #     result.append(group_results_list)
        return result

    def compute_result_and_split_into_output_elements(self, input_tensor):

        convolution_result = self.compute_convolution_result(input_tensor)
        # print(">>> compute_result_and_split_into_output_elements - convolution_result.size(): " +
        #      str(convolution_result.size()))

        if self.number_of_groups > 1:
            return self.split_result_into_output_elements_for_multiple_group_computation(convolution_result)
        else:
                return self.split_result_into_output_elements_for_single_group_computation(convolution_result)

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
