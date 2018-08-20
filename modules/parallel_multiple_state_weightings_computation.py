import torch.nn as nn
import torch.nn.functional as F
from modules.multi_dimensional_rnn import StateUpdateBlock
from torch.nn.modules.module import Module
import torch
from modules.inside_model_gradient_clipping import InsideModelGradientClamping
from modules.gradient_clamped_module import GradientClampedModule
from util.tensor_utils import TensorUtils


class ParallelMultipleStateWeightingsComputationBase(Module):
    def __init__(self, hidden_states_size: int,
                 output_states_size: int,
                 parallel_convolution: nn.Conv1d,
                 clamp_gradients: bool,
                 use_dropout: bool,
                 training: bool):
        super(ParallelMultipleStateWeightingsComputationBase, self).__init__()
        self.hidden_states_size = hidden_states_size
        self.output_states_size = output_states_size
        self.parallel_convolution = parallel_convolution
        self.clamp_gradients = clamp_gradients
        self.use_dropout = use_dropout
        self.training = training

    def get_number_of_paired_input_weightings(self):
        return sum(self.number_of_paired_input_weightings_per_group)

    def get_state_convolutions_as_list(self):
        return list([self.parallel_convolution])

    # When testing the model, training should be set to false
    def set_training(self, training):
        self.training = training

    def get_result_range_start_index(self, result_element_index):
        return self.hidden_states_size * result_element_index

    def get_result_range_end_index(self, result_element_index):
        return self.hidden_states_size * (result_element_index + 1)

    def compute_convolution_result(self, input_tensor: torch.Tensor):
        if self.use_dropout:
                # print("Applying dropout...")
                # TODO: which probability to use for dropout?
                result = F.dropout(self.parallel_convolution(input_tensor), p=0.2, training=self.training)
                return result
        result = self.parallel_convolution(input_tensor)
        return result

    """
    Compute the previous state columns for all groups from a list of previous state columns,
    by repeating each previous_state_column_i  N_i times, where N_i is the number of paired input
    weightings for the group of previous_state_column_i. By setting the groups of the convolution
    to the total number of paired input weightings, this allows to compute the results of 
    multiple convolutions from multiple input states with different numbers of output states.
    The trick of repeating the previous state columns multiple times in combination with the 
    "over-grouping" of the convolution in this way is necessary, because with "normal grouping"
    (i.e. the number of convolution groups is the number of previous states) it would not be possible
    to have different numbers of paired input weightings for the different states.  
    """
    def compute_previous_state_column_all_groups(self, previous_state_columns: list):
        cat_list = list([])
        for index, previous_state_column in enumerate(previous_state_columns):
            number_of_paired_weightings_group = self.number_of_paired_input_weightings_per_group[index]
            # print("number_of_paired_weightings_group: " + str(number_of_paired_weightings_group))
            previous_state_column_repeated_for_number_or_paired_weigthings_group = \
                list([previous_state_column] * number_of_paired_weightings_group)
            cat_list.extend(previous_state_column_repeated_for_number_or_paired_weigthings_group)
            # print("len(cat_list): " + str(len(cat_list)))
        # Concatenate the repeated states list over the channel dimension
        # print("final len(cat_list): " + str(len(cat_list)))
        # print("cat_list: " + str(cat_list))
        previous_state_column_all_groups = torch.cat(cat_list, 1)
        return previous_state_column_all_groups

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def create_second_pair_element_shifted_pair_list(convolution_result_pairs: list, clamp_gradients: bool):
        result = list([])

        for result_pair in convolution_result_pairs:
            pair_element_one = result_pair[0]
            # pair_element_two = result_pair[1]
            # print("pair_element_two: " + str(pair_element_two))
            # The second pair element is shifted, so that the right elements are combined
            # for multi-dimensional RNN/LSTM computation

            # Slow
            # pair_two_element_shifted = StateUpdateBlock.get_shifted_column(result_pair[1], self.hidden_states_size)
            # summed_values = pair_element_one + pair_two_element_shifted

            # Faster
            pair_two_element_shifted = StateUpdateBlock.get_shifted_column_fast(result_pair[1],
                                                                                clamp_gradients)
            pair = tuple((pair_element_one, pair_two_element_shifted))
            result.append(pair)
        return result

    def split_convolution_result_subtensor_into_output_pairs(self, convolution_result_subtensor):
        convolution_result_chunks = torch.chunk(convolution_result_subtensor,
                                                self.get_number_of_paired_input_weightings() * 2, 1)

        result = list([])
        for i in range(0, self.get_number_of_paired_input_weightings()):
            pair_element_one = convolution_result_chunks[i * 2]
            pair_element_two = convolution_result_chunks[i * 2 + 1]
            pair = tuple((pair_element_one, pair_element_two))
            result.append(pair)

        # Old implementation using slicing
        # for i in range(0, self.get_number_of_paired_input_weightings()):
        #     range_begin = self.get_result_range_start_index(i * 2)
        #     range_end = self.get_result_range_end_index(i * 2)
        #     # print("range begin: " + str(range_begin) + " range end: " + str(range_end))
        #     pair_element_one = convolution_result[:, range_begin:range_end, :]
        #     range_begin = self.get_result_range_start_index(i * 2 + 1)
        #     range_end = self.get_result_range_end_index(i * 2 + 1)
        #     # print("range begin: " + str(range_begin) + " range end: " + str(range_end))
        #     pair_element_two = convolution_result[:, range_begin:range_end, :]
        #     pair = tuple((pair_element_one, pair_element_two))
        #     result.append(pair)
        return result


# This class optimizes the computation of multiple states computed
# using 1D convolutions that are computed from the same input, by
# computing them as a single convolution with more outputs, and then
# splitting the results.
# The states are computed as pairs, whereby the result of the second
# element of the pair needs to be shifted by one position.
# This is for the purpose of querying one original and one shifted element
# in MDLSTM computation: by shifting the result of the input convolution
# for the second hidden/memory state, all the other computations become
# simpler, since now the shifting has already been done so the same element
# from the original and shifted output can be combined, rather than explicitly
# shifting the querying index for getting the shifted result.
class ParallelMultipleStateWeightingsComputation(ParallelMultipleStateWeightingsComputationBase):
    def __init__(self, hidden_states_size: int,
                 number_of_paired_input_weightings_per_group: list,
                 output_states_size: int,
                 parallel_convolution: nn.Conv1d,
                 clamp_gradients: bool,
                 use_dropout: bool,
                 training: bool):
        super(ParallelMultipleStateWeightingsComputation, self).__init__(
            hidden_states_size,
            output_states_size, parallel_convolution, clamp_gradients, use_dropout,
            training)
        self.number_of_paired_input_weightings_per_group = number_of_paired_input_weightings_per_group

    @staticmethod
    def create_parallel_multiple_state_weighting_computation(hidden_states_size: int,
                                                             number_of_paired_input_weightings: int,
                                                             clamp_gradients: bool,
                                                             use_dropout: bool):
        output_states_size = hidden_states_size * number_of_paired_input_weightings * 2

        # parallel_convolution = nn.Conv1d(hidden_states_size, output_states_size, 1)

        parallel_convolution = nn.Conv1d(hidden_states_size, output_states_size, 1)

        if clamp_gradients:
            parallel_convolution = GradientClampedModule(parallel_convolution)


        # Xavier weight initialization
        torch.nn.init.xavier_uniform_(parallel_convolution.weight)

        return ParallelMultipleStateWeightingsComputation(hidden_states_size, list([number_of_paired_input_weightings]),
                                                          output_states_size, parallel_convolution, clamp_gradients,
                                                          use_dropout,
                                                          True)

    @staticmethod
    def create_parallel_multiple_state_weighting_computation_multiple_groups(
            hidden_states_size: int, number_of_paired_input_weightings_per_group: list,
            clamp_gradients: bool, use_dropout: bool):
        number_of_paired_input_weightings = sum(number_of_paired_input_weightings_per_group)
        input_states_size = number_of_paired_input_weightings * hidden_states_size
        output_states_size = hidden_states_size * number_of_paired_input_weightings * 2

        # parallel_convolution = nn.Conv1d(hidden_states_size, output_states_size, 1)

        # print("create_parallel_multiple_state_weighting_computation_multiple_groups - \n"
        #       + "hidden_states_size: " + str(hidden_states_size) +
        #       " number_of_paired_input_weightings_per_group: " + str(number_of_paired_input_weightings_per_group) +
        #       "  output_states_size: " + str(output_states_size))
        parallel_convolution = nn.Conv1d(input_states_size, output_states_size, 1,
                                         groups=number_of_paired_input_weightings)

        if clamp_gradients:
            parallel_convolution = GradientClampedModule(parallel_convolution)

        # Xavier weight initialization
        torch.nn.init.xavier_uniform_(parallel_convolution.weight)

        return ParallelMultipleStateWeightingsComputation(hidden_states_size,
                                                          number_of_paired_input_weightings_per_group,
                                                          output_states_size, parallel_convolution, clamp_gradients,
                                                          use_dropout,
                                                          True)

    def get_paired_input_weightings_output_size(self):
        result = self.get_number_of_paired_input_weightings() * self.hidden_states_size * 2
        return result

    # How to do dropout in pytorch:
    # https://discuss.pytorch.org/t/dropout-functional-api-advantages-disadvantages/181/4
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    # Where to apply dropout:
    # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
    def compute_convolution_result_and_apply_mask(self, previous_state_column: torch.Tensor, mask: torch.Tensor):
        result = self.compute_convolution_result(previous_state_column)

        if self.clamp_gradients:
            # print("ParallelMultipleStateWeightingsComputation - register gradient clamping...")
            # Create a 1d convolution with clamping of the gradient
            result = InsideModelGradientClamping.\
                register_gradient_clamping_default_clamping_bound(result,
                                                                  "parallel_multiple_state_weightings_Computation",
                                                                  mask)

        # It is necessary to mask the non-valid entries in the convolution result. If this
        # is not done, then the results will be "incorrect" and also when using examples packing,
        # the first row in the packed matrix will be treated differently from the first rows
        # of other examples under vertical row separators.
        # For this reason, we must mask not only the states computed for the next iteration
        # during MDLSTM computation but also for the convolution computation the entries that
        # are not valid
        if not(mask is None):
            result = TensorUtils.apply_binary_mask(result, mask)

        # print("compute_convolution_result - result.size():" + str(result.size()))

        return result

    def compute_result_and_split_into_output_pairs(self, previous_state_column, mask: torch.Tensor):

        # print("compute_result_and_split_into_output_pairs - previous_state_column: " + str(previous_state_column))

        convolution_result = self.compute_convolution_result_and_apply_mask(previous_state_column, mask)
        return self.split_convolution_result_subtensor_into_output_pairs(convolution_result)

    def compute_result_and_split_into_pairs_with_second_pair_element_shifted(self, previous_state_column,
                                                                             mask: torch.Tensor=None):
        convolution_result_pairs = self.compute_result_and_split_into_output_pairs(previous_state_column, mask)
        return ParallelMultipleStateWeightingsComputationBase.\
            create_second_pair_element_shifted_pair_list(convolution_result_pairs, self.clamp_gradients)

    def compute_result_and_split_into_pairs_with_second_pair_element_shifted_multiple_groups(
            self, previous_state_columns, mask: torch.Tensor = None):
        # print("ParallelMultipleStateWeightingsComputation \n" +
        #      "compute_result_and_split_into_pairs_with_second_pair_element_shifted_multiple_groups...")
        previous_state_column_all_groups = self.compute_previous_state_column_all_groups(previous_state_columns)
        # print("previous_state_column_all_groups.size(): " + str(previous_state_column_all_groups.size()))
        return self.compute_result_and_split_into_pairs_with_second_pair_element_shifted(
            previous_state_column_all_groups, mask)

    @staticmethod
    def compute_summed_outputs_every_pair_static(convolution_result_pairs):
        # Original implementation
        result = list([])
        for result_pair in convolution_result_pairs:
            pair_element_one = result_pair[0]
            pair_two_element_shifted = result_pair[1]
            # print("pair two element shifted: " + str(pair_two_element_shifted))
            summed_values = pair_element_one + pair_two_element_shifted
            result.append(summed_values)
        return result

        # # Implementation with parallelized summation using stacking
        # result_pairs_first_elements = list([])
        # result_pairs_second_elements = list([])
        #
        # for result_pair in convolution_result_pairs:
        #     pair_element_one = result_pair[0]
        #     # print("pair_element_one.size(): " + str(pair_element_one.size()))
        #     pair_two_element_shifted = result_pair[1]
        #     # print("pair_two_element_shifted.size(): " + str(pair_two_element_shifted.size()))
        #     result_pairs_first_elements.append(pair_element_one)
        #     result_pairs_second_elements.append(pair_two_element_shifted)
        # first_stack = torch.stack(result_pairs_first_elements, 0)
        # second_stack = torch.stack(result_pairs_second_elements, 0)
        # summed_stacks = first_stack + second_stack
        # result_with_extra_dim = torch.split(summed_stacks, 1, 0)
        # result = list([])
        # for element in result_with_extra_dim:
        #     result.append(element.squeeze(0))
        # # print("result[0].size(): " + str(result[0].size()))
        # return result

    # This method :
    # 1. Computes the shared convolution over the previous_state_column
    # 2. Splits the output of the convolution with dimension of
    #    [batch_size, image_height, hidden_states_size * number_of_paired_input_weightings * 2]
    #    into pairs, each pair containing a part of the output of size
    #    [batch_size, image_height, hidden_states_size]
    #    the two pair elements correspond to a weighting of the first and second
    #    previous state respectively (to get the second previous state the results
    #    still need to be shifted by one position)
    # 3. Shift the second element of each pair one row down, and sum it with the first
    #    element to get the final output for each pair. Return the list of these
    #    results, which has number_of_paired_input_weightings elements
    def compute_summed_outputs_every_pair(self, previous_state_column,  mask: torch.Tensor):
        convolution_result_pairs = self.\
            compute_result_and_split_into_pairs_with_second_pair_element_shifted(previous_state_column, mask)
        return ParallelMultipleStateWeightingsComputation.compute_summed_outputs_every_pair_static(
            convolution_result_pairs)

    # This class extends Module so as to make sure that the parameters
    # are properly copied (to the right cuda device) when using nn.DataParallel(model)
    # and the to(device) method from  the Module base class
    # http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html
    # The class is not however meant to be used as a stand-alone Module, so forward
    # is not implemented
    def forward(self, x):
        raise NotImplementedError


class ParallelMultipleInputWeightingsComputation(ParallelMultipleStateWeightingsComputationBase):
    def __init__(self, hidden_states_size: int,
                 number_of_single_input_weightings_per_group: list,
                 output_states_size: int,
                 parallel_convolution: nn.Conv1d,
                 clamp_gradients: bool,
                 use_dropout: bool,
                 training: bool):
        super(ParallelMultipleInputWeightingsComputation, self).__init__(
            hidden_states_size,
            output_states_size, parallel_convolution, clamp_gradients, use_dropout,
            training)
        self.number_of_single_input_weightings_per_group = number_of_single_input_weightings_per_group

    @staticmethod
    def create_parallel_multiple_input_weighting_computation(input_channels: int,
            hidden_states_size: int, number_of_single_input_weightings_per_group: list,
            clamp_gradients: bool, use_dropout: bool):
        number_of_single_input_weightings = sum(number_of_single_input_weightings_per_group)

        # FIXME: The problem is that the number of channels of states and inputs is not the same
        # This makes computing both with the same convolution actually very tricky
        input_states_size = input_channels * number_of_single_input_weightings
        output_states_size = hidden_states_size * number_of_single_input_weightings

        # parallel_convolution = nn.Conv1d(hidden_states_size, output_states_size, 1)

        # print("create_parallel_multiple_state_weighting_computation_multiple_groups - \n"
        #       + "hidden_states_size: " + str(hidden_states_size) +
        #       "  output_states_size: " + str(output_states_size))
        number_of_groups = number_of_single_input_weightings

        # print("number of groups: " + str(number_of_groups))
        # print("input states size: " + str(input_states_size))
        # print("output states size: " + str(output_states_size))
        parallel_convolution = nn.Conv1d(input_states_size, output_states_size, 1,
                                         groups=number_of_groups)

        if clamp_gradients:
            parallel_convolution = GradientClampedModule(parallel_convolution)

        # Xavier weight initialization
        torch.nn.init.xavier_uniform_(parallel_convolution.weight)

        return ParallelMultipleInputWeightingsComputation(
            hidden_states_size, number_of_single_input_weightings_per_group,
            output_states_size, parallel_convolution, clamp_gradients,
            use_dropout, True)

    def get_number_of_single_input_weightings(self):
        return sum(self.number_of_single_input_weightings_per_group)

    def get_single_input_weightings_output_size(self):
        result = self.get_number_of_single_input_weightings() * self.hidden_states_size
        return result

    def compute_result_and_split_into_output_singletons(self, previous_state_column):

        # print("compute_result_and_split_into_output_pairs - previous_state_column: " + str(previous_state_column))

        input_convolution_result_input_sub_tensor = \
            self.compute_convolution_results(previous_state_column)
        input_convolution_result_singletons = self.split_convolution_result_subtensor_into_single_outputs(
            input_convolution_result_input_sub_tensor)
        return input_convolution_result_singletons

    def split_convolution_result_subtensor_into_single_outputs(self, convolution_result_subtensor):
        convolution_result_chunks = torch.chunk(convolution_result_subtensor,
                                                self.get_number_of_single_input_weightings(), 1)
        return convolution_result_chunks

    def compute_convolution_results(self, input_tensor: torch.Tensor):
        # print("input_tensor.size(): " + str(input_tensor.size()))
        result = self.compute_convolution_result(input_tensor)
        # print("result.size(): " + str(result.size()))

        if self.clamp_gradients:
            # print("ParallelMultipleStateWeightingsComputation - register gradient clamping...")
            # Create a 1d convolution with clamping of the gradient
            result = InsideModelGradientClamping. \
                register_gradient_clamping_default_clamping_bound(result,
                                                                  "parallel_multiple_state_weightings_Computation",
                                                                  None)

        return result

    def compute_result_and_split_into_columns_multiple_groups(
            self, input_columns):
        # print("ParallelMultipleStateWeightingsComputation \n" +
        #      "compute_result_and_split_into_pairs_with_second_pair_element_shifted_multiple_groups...")
        previous_state_column_and_input_column_all_groups = self.\
            compute_input_column_all_groups(input_columns)
        # print("previous_state_column_and_input_column_all_groups.size(): " +
        #       str(previous_state_column_and_input_column_all_groups.size()))
        return self.compute_result_and_split_into_output_singletons(
            previous_state_column_and_input_column_all_groups)

    def compute_input_column_all_groups(self, input_columns: list):
        cat_list = list([])

        for index, input_column in enumerate(input_columns):
            number_of_single_weightings_group = self.number_of_single_input_weightings_per_group[index]
            # print("number_of_single_weightings_group: " + str(number_of_single_weightings_group))
            input_column_repeated_for_number_or_single_weigthings_group = \
                list([input_column] * number_of_single_weightings_group)
            cat_list.extend(input_column_repeated_for_number_or_single_weigthings_group)
            # print("len(cat_list): " + str(len(cat_list)))
        # Concatenate the repeated states list over the channel dimension
        # print("final len(cat_list): " + str(len(cat_list)))
        # print("cat_list: " + str(cat_list))

        # Concatenate the repeated states list over the channel dimension
        # print("final len(cat_list): " + str(len(cat_list)))
        # print("cat_list: " + str(cat_list))
        previous_state_column_all_groups = torch.cat(cat_list, 1)
        # print(" previous_state_column_all_groups.size(): "
        #       + str( previous_state_column_all_groups.size()))
        return previous_state_column_all_groups

    def forward(self, x):
            raise NotImplementedError

