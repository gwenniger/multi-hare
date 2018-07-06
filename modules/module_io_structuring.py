from util.tensor_utils import TensorUtils
import torch

class ModuleIOStructuring:

    @staticmethod
    def compute_paired_lists_multiples(int_list_one, int_list_two):
        result = list([])
        for element_one, element_two in zip(int_list_one, int_list_two):
            multiple = element_one * element_two
            # print("compute_paired_lists_multiples - element one: " +
            #       str(element_one) + " element two: "
            #       + str(element_two))
            # print("compute_paired_lists_multiples - multiple: " + str(multiple))
            result.append(multiple)
        return result

    # Extract the summed rows from a chunk that consists multiple rows
    @staticmethod
    def extract_summed_rows_from_chunk_with_concatenated_rows(chunk_multiple_rows, number_of_rows,
                                                              number_of_columns):
        # print("chunk multiple rows.size(): " + str(chunk_multiple_rows.size()))

        # print("number of columns: " + str(number_of_columns) +
        #      " number of rows: " + str(number_of_rows))
        # Notice the dimension to split on is 1, as for example we have
        # chunk multiple rows.size(): torch.Size([1, 14, 80])
        # That is, the last dimension goes over classes, the first one is
        # always 1, and the second dimension goes over the width.
        # Therefore we have to split on dim=1 using the number_of_columns
        # for the tensor containing the horizontally-concatenated
        # row activations
        rows = torch.split(chunk_multiple_rows, number_of_columns, dim=1)
        if len(rows) != number_of_rows:
            raise RuntimeError("Error in split: expected " + str(number_of_rows)
                               + "rows but got: " + str(len(rows)))

        summed_rows = TensorUtils.sum_list_of_tensors(rows)
        # print("summed_rows.size(): " + str(summed_rows.size()))
        return summed_rows

    @staticmethod
    def extract_activation_chunks(examples_activation_heights, examples_activation_widths,
                                  class_activations_resized_temp):
        examples_activation_height_times_width = ModuleIOStructuring. \
            compute_paired_lists_multiples(examples_activation_heights, examples_activation_widths)
        chunks_multiple_rows = torch.split(class_activations_resized_temp,
                                           examples_activation_height_times_width, 1)
        chunks = list([])

        for index in range(0, len(chunks_multiple_rows)):
            chunk_multiple_rows = chunks_multiple_rows[index]
            number_of_rows = examples_activation_heights[index]
            number_of_columns = examples_activation_widths[index]
            summed_rows = ModuleIOStructuring. \
                extract_summed_rows_from_chunk_with_concatenated_rows(chunk_multiple_rows, number_of_rows,
                                                                      number_of_columns)
            chunks.append(summed_rows)

        return chunks

    # This debugging method checks that de-chunking the chunked tensor recovers the original
    @staticmethod
    def check_dechunking_chunked_tensor_list_recovers_original(tensor_list_chunking, original_tensor_list,
                                                       input_chunked):
        input_dechunked = tensor_list_chunking.dechunk_block_tensor_concatenated_along_batch_dimension(
            input_chunked)
        if not TensorUtils.tensors_lists_are_equal(original_tensor_list, input_dechunked):
            for index in range(0, len(original_tensor_list)):
                print("original[" + str(index) + "].size()" + str(original_tensor_list[index].size()))

            for index in range(0, len(input_dechunked)):
                print("input_dechunked[" + str(index) + "].size()" + str(input_dechunked[index].size()))

            TensorUtils.find_equal_slices_over_batch_dimension(input_chunked)

            raise RuntimeError("Error: original and de-chunked chunked are not the same")

    # This debugging method looks if it can find equal rows in the activation
    # (which in general should typically not happen)
    @ staticmethod
    def check_activation_rows_are_not_equal(activation_rows):
        # For debugging
        print("activation rows sizes after splitting: ")
        last_activation_row = activation_rows[0]
        for activation_row in activation_rows:
            print(str(activation_row.size()))
            if TensorUtils.tensors_are_equal(last_activation_row, activation_row):
                print(">>> WARNING: activation rows are equal")

    @staticmethod
    def compute_non_padding_activation_widths(input_examples_list, width_required_per_network_column):
        result = list([])
        for input_example in input_examples_list:
            number_of_outputs_for_example = int(input_example.size(2) / width_required_per_network_column)
            result.append(number_of_outputs_for_example)
        return result

    @staticmethod
    def extract_and_concatenate_nonpadding_parts_activations(input_examples_list, activations,
                                                             width_required_per_network_column):

        non_padding_activation_widths = ModuleIOStructuring.\
            compute_non_padding_activation_widths(input_examples_list, width_required_per_network_column)
        number_of_examples = activations.size(0)

        non_padding_parts_for_cat = list([])
        for example_index in range(0, number_of_examples):
            non_padding_activation_width = non_padding_activation_widths[example_index]
            print("non_padding_activation_width: " + str(non_padding_activation_width))
            non_padding_example_activations = activations[example_index, :, :, 0:non_padding_activation_width]
            print("non_padding_example_activations.size(): " + str(non_padding_example_activations.size()))
            non_padding_example_activation_rows = torch.split(non_padding_example_activations, 1, 1)
            for element in non_padding_example_activation_rows:
                print("non_padding_example_activation_row.size(): " + str(element.size()))
            non_padding_parts_for_cat.extend(non_padding_example_activation_rows)
        result = torch.cat(non_padding_parts_for_cat, 2)
        print("extract_and_concatenate_nonpadding_parts_activations - result.size(): " + str(result.size()))
        # Remove superfluous height dimension: output will be of dimension:  total_width * no_channels
        result = result.squeeze(1)
        result = result.transpose(0, 1)
        return result


