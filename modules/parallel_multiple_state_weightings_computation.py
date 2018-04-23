import torch.nn as nn
from modules.multi_dimensional_rnn import StateUpdateBlock


class ParallelMultipleStateWeightingsComputation:
    def __init__(self, hidden_states_size: int,
                 number_of_paired_input_weightings,
                 output_states_size,
                 parallel_convolution,):
        self.hidden_states_size = hidden_states_size
        self.number_of_paired_input_weightings = number_of_paired_input_weightings
        self.output_states_size = output_states_size
        self.parallel_convolution = parallel_convolution

    @staticmethod
    def create_parallel_multiple_state_weighting_computation(hidden_states_size: int,
                                                             number_of_paired_input_weightings: int):
        output_states_size = hidden_states_size * number_of_paired_input_weightings * 2

        parallel_convolution = nn.Conv1d(hidden_states_size, output_states_size, 1)
        return ParallelMultipleStateWeightingsComputation(hidden_states_size, number_of_paired_input_weightings,
                                                          output_states_size, parallel_convolution)

    def compute_convolution_result(self, previous_state_column):
        result = self.parallel_convolution(previous_state_column)
        return result

    def get_result_range_start_index(self, result_element_index):
        return self.hidden_states_size * result_element_index

    def get_result_range_end_index(self, result_element_index):
        return self.hidden_states_size * (result_element_index + 1)

    def compute_result_and_split_into_output_pairs(self, previous_state_column):
        result = list([])
        convolution_result = self.compute_convolution_result(previous_state_column)
        for i in range(0, self.number_of_paired_input_weightings):

            pair_element_one = \
                convolution_result[:, self.get_result_range_start_index(i*2):
                                   self.get_result_range_end_index(i*2), :]
            pair_element_two = \
                convolution_result[:, self.get_result_range_start_index(i * 2 + 1):
                                   self.get_result_range_end_index(i * 2 + 1), :]
            pair = tuple((pair_element_one, pair_element_two))
            result.append(pair)
        return result

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
    def compute_summed_outputs_every_pair(self, previous_state_column):
        result = list([])
        convolution_result_pairs = self.compute_result_and_split_into_output_pairs(previous_state_column)
        for result_pair in convolution_result_pairs:
            pair_element_one = result_pair[0]
            pair_element_two = result_pair[1]
            # print("pair_element_two: " + str(pair_element_two))
            # The second pair element is shifted, so that the right elements are combined
            # for multi-dimensional RNN/LSTM computation

            # Slow
            # pair_two_element_shifted = StateUpdateBlock.get_shifted_column(result_pair[1], self.hidden_states_size)
            # summed_values = pair_element_one + pair_two_element_shifted

            # Faster
            pair_two_element_shifted = StateUpdateBlock.get_shifted_column_fast(result_pair[1], self.hidden_states_size)
            summed_values = pair_element_one + pair_two_element_shifted

            # Not really faster but simpler and about the same speed
            # summed_values = pair_element_one.clone()
            # summed_values[:, :, 1:] = summed_values[:, :, 1:] +
            #   pair_element_two[:, :, 0: pair_element_two.size(2) - 1]

            result.append(summed_values)
        return result

    def get_state_convolutions_as_list(self):
        return list([self.parallel_convolution])
