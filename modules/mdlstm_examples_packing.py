from modules.size_two_dimensional import SizeTwoDimensional
from sortedcontainers.sortedlist import SortedList
# See: https://stackoverflow.com/questions/31493603/
# alternative-to-pythons-sort-for-inserting-into-a-large-list-and-keeping-it
# http://www.grantjenks.com/docs/sortedcontainers/
# Sorted containers will be used to efficiently find the closest elements that
# that are smaller than some value
import torch
from util.image_input_transformer import ImageInputTransformer
from util.tensor_utils import TensorUtils
from util.utils import Utils
import util.image_visualization
from util.tensor_flipping import TensorFlipping


class IndexedExampleSize:

    def __init__(self,  original_example_index: int, example_size: SizeTwoDimensional):
        self.original_example_index = original_example_index
        self.example_size = example_size

    @staticmethod
    def create_indexed_example_size(original_example_index: int, height: int, width: int):
        size = SizeTwoDimensional.create_size_two_dimensional(height, width)

        return IndexedExampleSize(original_example_index, size)

    def get_mdlstm_skewed_image_width(self):
        return ImageInputTransformer.get_skewed_images_width(self.example_size.height,
                                                             self.example_size.width)

    # We need to implement the comparable operators in order to be able
    # to make use of sortedcontainers
    # http://gerg.ca/blog/post/2012/python-comparison/
    def __eq__(self, other):
        return self.example_size == other.example_size and \
               self.original_example_index == other.original_example_index

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return self.example_size.width > other.example_size.width

    def __lt__(self, other):
        return self.example_size.width < other.example_size.width

    def __ge__(self, other):
        return (self > other) or (self.example_size.width == other.example_size.width)

    def __le__(self, other):
        return (self < other) or (self.example_size.width == other.example_size.width)

   #http://gerg.ca/blog/post/2012/python-comparison/

    def __str__(self):
        result = "<I.E.Size> "
        result += "original_example_index: " + str(self.original_example_index) + " "
        result += "example_size: " + str(self.example_size) + " "
        result += "mdlstm-width: " + str(self.get_mdlstm_skewed_image_width()) + " "
        result += "</I.E.Size>"
        return result

    def __repr__(self):
        return self.__str__()


class MDLSTMExamplesPacking:

    def __init__(self, packed_examples,
                 original_example_index_to_packed_index_table,
                 max_example_width: int,
                 example_separator_width: int):
        self.packed_examples = packed_examples
        self.original_example_index_to_packed_index_table = original_example_index_to_packed_index_table
        self.max_example_width = max_example_width
        self.example_separator_width = example_separator_width

    @staticmethod
    def created_mdlstm_examples_packing(examples_list: list, example_separator_width: int):
        packed_examples, max_example_width = MDLSTMExamplesPacking.get_packed_examples(examples_list, example_separator_width)
        original_example_index_to_packed_index_table = \
            MDLSTMExamplesPacking.create_original_example_index_to_packed_index_table(packed_examples, examples_list)
        return MDLSTMExamplesPacking(packed_examples,
                                     original_example_index_to_packed_index_table,
                                     max_example_width, example_separator_width)

    @staticmethod
    def get_packed_examples(examples_list: list, example_separator_width: int):
        example_sizes_list = MDLSTMExamplesPacking. \
            get_example_sizes_from_examples_list(examples_list)
        height_grouped_examples_table = MDLSTMExamplesPacking.\
            get_height_grouped_examples_table(example_sizes_list)
        max_example_width = MDLSTMExamplesPacking.\
            get_maximum_example_width(example_sizes_list)

        result = list([])

        for height in height_grouped_examples_table:
            examples_for_height = height_grouped_examples_table[height]
            packed_examples_list = MDLSTMExamplesPacking.greedy_pack_examples_of_same_height(
                examples_for_height, max_example_width, example_separator_width)
            result.extend(packed_examples_list)

        return result, max_example_width

    @staticmethod
    def create_original_example_index_to_packed_index_table(packed_examples, examples_list):
        original_index_to_reordered_index_table = [None] * len(examples_list)

        reordered_index = 0
        for packed_examples_row in packed_examples:
            for indexed_example_size in packed_examples_row:
                original_index = indexed_example_size.original_example_index
                original_index_to_reordered_index_table[original_index] = reordered_index
                reordered_index += 1
        return original_index_to_reordered_index_table


    @staticmethod
    def get_mdlstm_computation_rows_skewing_overhead(examples_height):
        skewing_overhead = examples_height - 1
        return skewing_overhead

    @staticmethod
    def get_packed_example_widths(packed_examples_row: list):
        result = 0
        for indexed_example_size in packed_examples_row:
            result += indexed_example_size.example_size.width
        return  result

    @staticmethod
    def get_packed_example_widths_plus_skewing_overhead(packed_examples_row: list):
        result = MDLSTMExamplesPacking.get_packed_example_widths(packed_examples_row)
        examples_height = packed_examples_row[0].example_size.height
        skewing_overhead = MDLSTMExamplesPacking.get_mdlstm_computation_rows_skewing_overhead(examples_height)
        result += skewing_overhead
        return result

    def get_packed_example_widths_plus_separator_overhead(self, packed_examples_row: list):
        result = MDLSTMExamplesPacking.get_packed_example_widths(packed_examples_row)
        result += (len(packed_examples_row) - 1) * self.example_separator_width
        return result

    """
    Compute the total width of a packed examples row, including the overhead of 
    skewing the rows for computation, and the overhead of one pixel in-between 
    example separators
    """
    def get_packed_example_widths_total(self, packed_examples_row: list):
        return MDLSTMExamplesPacking.get_packed_example_widths_plus_skewing_overhead(packed_examples_row) + \
               (len(packed_examples_row) - 1) * self.example_separator_width

    @staticmethod
    def get_packed_examples_row_height(packed_examples_row: list):
        return packed_examples_row[0].example_size.height

    @staticmethod
    def print_packed_examples_row(packed_examples_row: list):
        summed_example_widths = MDLSTMExamplesPacking.get_packed_example_widths(packed_examples_row)
        print("packed examples row - height: " +
              str(MDLSTMExamplesPacking.get_packed_examples_row_height(packed_examples_row)) +
              " summed_example_widths: "
              + str(summed_example_widths) + "\n - example sizes: " + str(packed_examples_row))

    """
    This method greedily packs examples of the same height into lists of multiple examples whose 
    summed widths, including an additional "example_separator_width" for each but the first example,
    must be smaller than maximum_example_width. 
    
    The filling of a row for the greedy packing is done by repeatedly finding the smallest example 
    amongst the remaining list of examples, that still fits in the remaining space for the row.
    After this additional example is added, the remaining space for the row is updated by subtracting
    the width of the just added example and the example_separator_width.
    This is repeated until no small enough example can be found anymore to further fill up the row.
    The filled row is then added to the result, and a new, empty row is created and step-by-step
    filled up in the same way.
    This process continues until all examples from examples_list have been added to rows.
    
    For efficiently finding the closest example smaller or equal than the remaining width, 
    this method uses the SortedList class from the sortedcontainers library. Specifically, it uses the method
    bisect_right, to find the index where the query item should be inserted and from that position infers the 
    index of the just smaller example (by subtracting 1).
    
    The hope is that, while not optimal, this greedy packing will find a pretty efficient packing of the examples, 
    that will drastically reduce the amount of padding over the alternative of not using the packing.
    """
    @staticmethod
    def greedy_pack_examples_of_same_height(examples_list: list, maximum_example_width: int,
                                            example_separator_width: int):
        sorted_list = SortedList(examples_list)

        # print("sorted_list: " + str(sorted_list))

        examples_height = examples_list[0].example_size.height

        # Already subtract the width used for horizontally skewing the rows for
        # MDLSTM computation, when calculating the effective horizontal space
        # available given the height of the examples
        horizontal_space_per_row = maximum_example_width - \
            MDLSTMExamplesPacking.get_mdlstm_computation_rows_skewing_overhead(examples_height)

        result = list([])
        result_row = list([])
        space_remaining_current_row = horizontal_space_per_row

        while len(sorted_list) > 0:
            # print("len(sorted_list): " + str(len(sorted_list)))
            # print("sorted_list: " + str(sorted_list))
            # print("space_remaining_current_row: " + str(space_remaining_current_row))

            query_example = IndexedExampleSize.create_indexed_example_size(-1, examples_height,
                                                                           space_remaining_current_row)
            # print("query_example: " + str(query_example))
            #
            # print("query_example: " + str(query_example))
            # print("sorted_list: " + str(sorted_list))

            # bisect_rigth -1 works, but disturbs the natural order (it does not preserve the order
            # when changing the order is not required for equal width elements)
            # largest_element_smaller_or_equal_than_max_width_index = sorted_list.bisect_right(query_example) - 1

            # With bisect_left we get the smallest element equal or larger
            smallest_element_larger_or_equal_than_max_width_index = sorted_list.bisect_left(query_example)
            # But we want the largest element equal or smaller

            # print("smallest_element_larger_or_equal_than_max_width_index: " +
            #       str(smallest_element_larger_or_equal_than_max_width_index))

            # If the smallest_element_larger_or_equal_than_max_width_index is outside of the array
            # (i.e. it does not exist) : take the position to the left, which will exist and be
            # smaller
            if MDLSTMExamplesPacking.index_outside_list_on_right(smallest_element_larger_or_equal_than_max_width_index,
                                                                 sorted_list):
                largest_element_smaller_or_equal_than_max_width_index = \
                    smallest_element_larger_or_equal_than_max_width_index - 1
            # If the index is valid but the element at the index is larger: also
            # take the one to the left
            elif MDLSTMExamplesPacking.index_inside_list(smallest_element_larger_or_equal_than_max_width_index,
                                                         sorted_list) and MDLSTMExamplesPacking.\
                    width_element_at_index_larger_than_value(sorted_list,
                                                             smallest_element_larger_or_equal_than_max_width_index,
                                                             space_remaining_current_row):
                    # print("element is larger")
                    largest_element_smaller_or_equal_than_max_width_index = \
                        smallest_element_larger_or_equal_than_max_width_index - 1
            # The element is the same size, use that one
            else:
                # print("element is same size")
                largest_element_smaller_or_equal_than_max_width_index = \
                    smallest_element_larger_or_equal_than_max_width_index

            # print("largest_element_smaller_or_equal_than_max_width_index: " +
            #       str(largest_element_smaller_or_equal_than_max_width_index))

            # No suitable smaller example is left
            if largest_element_smaller_or_equal_than_max_width_index < 0:
                result.append(result_row)
                result_row = list([])
                space_remaining_current_row = horizontal_space_per_row
                # print("Starting new result_row...")
            else:
                largest_element_smaller_than_max_width = sorted_list.pop(largest_element_smaller_or_equal_than_max_width_index)
                # print("largest_element_smaller_than_max_width: " + str(largest_element_smaller_than_max_width))
                result_row.append(largest_element_smaller_than_max_width)
                space_remaining_current_row -= largest_element_smaller_than_max_width.example_size.width

                if space_remaining_current_row + example_separator_width < 0:
                    raise RuntimeError("Error space remaining current row " +
                                       str(space_remaining_current_row) +
                                       " has become negative")
                # Reserve space for required separator if additional example is added
                space_remaining_current_row -= example_separator_width



        # Add the last result row
        result.append(result_row)

        return result

    @staticmethod
    def index_outside_list_on_right(index, elements_list):
        return index >= len(elements_list)

    @staticmethod
    def index_inside_list(index, elements_list):
        return (index >= 0) and (index < len(elements_list))

    @staticmethod
    def width_element_at_index_larger_than_value(sorted_list, index: int, value: int):
        # print(" width_element_at_index_larger_than_value ")
        # print("   sorted_list[index].example_size.width:  " + str(sorted_list[index].example_size.width))
        # print("   value:  " + str(value))
        result = sorted_list[index].example_size.width > value
        # print("result: " + str(result))
        return result

    @staticmethod
    def get_maximum_example_width(example_sizes_list: list):
        result = 0
        for indexed_example_size in example_sizes_list:
            result = max(result, indexed_example_size. get_mdlstm_skewed_image_width())
        return result

    @staticmethod
    def get_height_grouped_examples_table(example_sizes_list: list):

        result = dict([])

        for indexed_example_size in example_sizes_list:

            example_size = indexed_example_size.example_size
            if example_size.height in result:
                examples_list = result[example_size.height]
            else:
                examples_list = list([])

            examples_list.append(indexed_example_size)
            result[example_size.height] = examples_list

        return result

    @staticmethod
    def get_example_sizes_from_examples_list(examples_list):

        result = list([])

        example_index = 0
        for example in examples_list:
            height = example.size(1)
            width = example.size(2)
            example_size = SizeTwoDimensional.create_size_two_dimensional(height,
                                                                          width)
            indexed_example_size = IndexedExampleSize(example_index, example_size)
            result.append(indexed_example_size)

            example_index += 1

        return result

    def get_non_padding_fraction(self):

        total_non_padding_pixels = 0
        total_pixels = 0

        for packed_examples_row in self.packed_examples:
            height = MDLSTMExamplesPacking.get_packed_examples_row_height(packed_examples_row)
            summed_widths = MDLSTMExamplesPacking.get_packed_example_widths(packed_examples_row)
            row_non_padding_pixels = height * summed_widths
            total_row_pixels = height * self.max_example_width

            total_non_padding_pixels += row_non_padding_pixels
            total_pixels += total_row_pixels

        return float(total_non_padding_pixels) / total_pixels

    def print_packed_examples_rows(self):
        for packed_examples_row in self.packed_examples:
            MDLSTMExamplesPacking.print_packed_examples_row(packed_examples_row)

    def create_horizontal_separator(self, example_unsqueezed_multi_directional):
        channels = example_unsqueezed_multi_directional.size(1)
        example_height = example_unsqueezed_multi_directional.size(2)
        device = example_unsqueezed_multi_directional.get_device()

        # print("create_horizontal_separator - device: " + str(device))
        # https://discuss.pytorch.org/t/creating-tensors-on-gpu-directly/2714
        with torch.cuda.device(device):
            result = torch.cuda.FloatTensor(1, channels,  example_height, self.example_separator_width).fill_(0)
        # print("create_horizontal_separator - result: " + str(result))
        return result

    def create_vertical_separator(self, skewed_packed_example_row_tensor):
        device = skewed_packed_example_row_tensor.get_device()
        with torch.cuda.device(device):
            # print("create_vertical_separator - device: " + str(device))
            return torch.cuda.FloatTensor(1, skewed_packed_example_row_tensor.size(1), 1, self.max_example_width).fill_(0)

    def create_vertical_mask_separator(self, example):
        device = example.get_device()
        with torch.cuda.device(device):
            # print("create_vertical_separator - device: " + str(device))
            return torch.cuda.FloatTensor(1, self.max_example_width).fill_(0)

    def create_extra_padding(self, row_cat_list, packed_examples_row):
        # Create and add extra padding needed to fill up the remaining columns
        # of the row up to self.max_example_width
        current_width = self.get_packed_example_widths_total(packed_examples_row)
        # print("current_width: " + str(current_width))
        # print("max_example_width: " + str(self.max_example_width))

        # Sanity check
        if current_width > self.max_example_width:
            raise RuntimeError("Error: current width " + str(current_width)
                               + " is greater than max_example_width "
                               + str(self.max_example_width))

        columns_extra_padding_required = self.max_example_width - current_width

        if columns_extra_padding_required == 0:
            return None

        channels = row_cat_list[0].size(1)
        height = row_cat_list[0].size(2)

        # print("create_extra_padding - channels: " + str(channels))
        # print("create_extra_padding - height: " + str(height))
        # print("create_extra_padding - columns_extra_padding_required: " + str(columns_extra_padding_required))
        device = row_cat_list[0].get_device()
        # print("device example 0: " + str(device))
        with torch.cuda.device(device):
            extra_padding = torch.cuda. \
                FloatTensor(1, channels, height,

                            columns_extra_padding_required).fill_(0)
        return extra_padding

    def create_mask_extra_padding(self, mask_row_cat_list, packed_examples_row):
        # Create and add extra padding needed to fill up the remaining columns
        # of the row up to self.max_example_width
        current_width = self.get_packed_example_widths_plus_skewing_overhead(packed_examples_row) + \
                        (len(packed_examples_row) - 1) * self.example_separator_width
        # print("current_width: " + str(current_width))
        # print("max_example_width: " + str(self.max_example_width))

        columns_extra_padding_required = self.max_example_width - current_width

        if columns_extra_padding_required == 0:
            return None

        height = mask_row_cat_list[0].size(0)

        # print("create_mask_extra_padding - height: " + str(height))
        # print("create_mask_extra_padding - columns_extra_padding_required: " + str(columns_extra_padding_required))
        device = mask_row_cat_list[0].get_device()
        # print("device example 0: " + str(device))
        with torch.cuda.device(device):
            extra_padding = torch.cuda. \
                FloatTensor(height, columns_extra_padding_required).fill_(0)
        return extra_padding

    def create_row_mask_packed_mdlstm_computation(self, packed_examples_row, device):

        height = packed_examples_row[0].example_size.height
        width = self.get_packed_example_widths_total(packed_examples_row)
        mask_tensor = torch.ones((height, width), out=None, dtype=torch.float,
                                 device=device)

        unskewed_image_width = width - height + 1

        # Take care of the image skewing at the beginning and end
        for row_number in range(0, height):
            first_valid_column_for_row = row_number
            last_valid_column_for_row = first_valid_column_for_row + unskewed_image_width
            mask_tensor[row_number, 0:first_valid_column_for_row] = 0
            mask_tensor[row_number, last_valid_column_for_row:width] = 0

        # Create diagonal example separators, that are parallel to the skewing
        # at the beginning and the end
        example_separator_index = -1
        for example_index in range(0, len(packed_examples_row) - 1):

            # Update the index for the next separator
            indexed_example_size = packed_examples_row[example_index]
            example_separator_index += indexed_example_size.example_size.width + 1

            for row_number in range(0, height):
                mask_tensor[row_number, example_separator_index + row_number] = 0

        return mask_tensor

    """
    This method efficiently skews all the packed row tensors of the same height 
    in one go, in parallel. This exploits the fact that adding the required amount
    of padding on the right of the un-skewed tensors, and then skewing them gives
    the same result as first adding the padding and then performing the skewing.
    Because of this, is tis possible to first pad all the un-skewed tensors of 
    the same height to the same width, then skew them all at once. This saves 
    out running the somewhat expensive skewing function for each of the packed rows 
    separately. 
    """
    def skew_parallel_vertically_pad_and_add_packed_row_tensors(self, same_height_packed_row_tensors,
                                                                result_cat_list):
        # print("skew_parallel_vertically_pad_and_add_packed_row_tensors...")
        stacked_same_height_packed_tensors = torch.cat(same_height_packed_row_tensors, 0)
        stacked_same_height_packed_tensors_skewed = ImageInputTransformer. \
            create_skewed_images_variable_four_dim(stacked_same_height_packed_tensors)
        # print("stacked_same_height_packed_tensors_skewed.size(): "
        #      + str(stacked_same_height_packed_tensors_skewed.size()))
        same_height_packed_tensors_skewed_list = torch.split(stacked_same_height_packed_tensors_skewed, 1, 0)
        # print("len(same_height_packed_tensors_skewed_list): " + str(len(same_height_packed_tensors_skewed_list)))
        for skewed_packed_example_row_tensor in same_height_packed_tensors_skewed_list:
            if len(result_cat_list) > 0:
                vertical_separator = self.create_vertical_separator(skewed_packed_example_row_tensor)
                # print("vertical_separator.size(): " + str(vertical_separator.size()))
                result_cat_list.append(vertical_separator)
            # print("skewed_packed_example_row_tensor.size(): " + str(skewed_packed_example_row_tensor.size()))
            result_cat_list.append(skewed_packed_example_row_tensor)

    @staticmethod
    def create_multi_directional_examples_stacked_on_channel_direction(example_unsqueezed,
                                                                       tensor_flippings: list):
        cat_list = list([])

        for tensor_flipping in tensor_flippings:
            if tensor_flipping is not None:
                # Flip example for computation to obtain the example with the right scanning
                # direction for Multi-Directional MDLSTM
                example_unsqueezed_flipped_for_current_direction = tensor_flipping.flip(example_unsqueezed)
            else:
                example_unsqueezed_flipped_for_current_direction = example_unsqueezed
            cat_list.append(example_unsqueezed_flipped_for_current_direction)
        return torch.cat(cat_list, 1)

    def create_vertically_and_horizontally_packed_examples_multiple_directions(self, examples: list,
                                                                               tensor_flippings: list):

        number_of_dimensions = TensorUtils.number_of_dimensions(examples[0])
        if number_of_dimensions != 3:
            raise RuntimeError("Error: expected an examples tensor with 3 "
                               "dimensions but got: " + str(number_of_dimensions))

        result_cat_list = list([])

        current_height = self.packed_examples[0][0].example_size.height
        same_height_packed_row_tensors = list([])

        for packed_examples_row in self.packed_examples:
            # print("create_vertically_and_horizontally_packed_examples - packed_examples_row: "
            #     + str(packed_examples_row))

            packed_row_height = packed_examples_row[0].example_size.height

            if packed_row_height != current_height:
                # Finished the rows of the previous height

                self.skew_parallel_vertically_pad_and_add_packed_row_tensors(same_height_packed_row_tensors,
                                                                             result_cat_list)
                same_height_packed_row_tensors = list([])

            row_cat_list = list([])
            for indexed_example_size in packed_examples_row:
                example_index = indexed_example_size.original_example_index
                example = examples[example_index]
                example_unsqueezed = example.unsqueeze(0)

                example_unsqueezed_flipped_for_multiple_directions_stacked =\
                    MDLSTMExamplesPacking.create_multi_directional_examples_stacked_on_channel_direction(
                        example_unsqueezed, tensor_flippings)

                if len(row_cat_list) > 0:
                    row_cat_list.append(self.create_horizontal_separator(
                        example_unsqueezed_flipped_for_multiple_directions_stacked))

                # print(" example_unsqueezed_flipped_for_multiple_directions_stacked.size(): " +
                #       str(example_unsqueezed_flipped_for_multiple_directions_stacked.size()))

                row_cat_list.append(example_unsqueezed_flipped_for_multiple_directions_stacked)

            extra_padding = self.create_extra_padding(row_cat_list, packed_examples_row)
            if extra_padding is not None:
                # print("extra_padding: " + str(extra_padding))
                row_cat_list.append(extra_padding)

            # for element in row_cat_list:
            #     print("row cat list element.size(): " + str(element.size()))
            catted_row_unskewed = torch.cat(row_cat_list, 3)
            same_height_packed_row_tensors.append(catted_row_unskewed)

            # print("len(row_cat_list): " + str(len(row_cat_list)))
            # print("row_cat_list: " + str(row_cat_list))
            # for element in row_cat_list:
            #    print("row_cat_list element.size(): " + str(element.size()))

            # print("catted_row_unskewed: " + str(catted_row_unskewed))

        # print("Add final same height tensors...")
        # Add final same height tensors
        self.skew_parallel_vertically_pad_and_add_packed_row_tensors(same_height_packed_row_tensors,
                                                                     result_cat_list)
        # print("len(result_cat_list): " + str(len(result_cat_list)))
        result = torch.cat(result_cat_list, 2)

        # print("create_vertically_and_horizontally_packed_examples_one_direction --- finished" +
        #       "result.size(): " + str(result.size()))

        # packed_examples_2d = result.squeeze(1)
        # packed_examples_2d = packed_examples_2d.squeeze(0)
        # util.image_visualization.imshow_tensor_2d(packed_examples_2d.cpu())

        return result

    def create_vertically_and_horizontally_packed_examples_mask_one_direction(self, examples: list):

        number_of_dimensions = TensorUtils.number_of_dimensions(examples[0])
        if number_of_dimensions != 3:
            raise RuntimeError("Error: expected an examples tensor with 3 "
                               "dimensions but got: " + str(number_of_dimensions))

        mask_result_cat_list = list([])

        for packed_examples_row in self.packed_examples:
            # print("create_vertically_and_horizontally_packed_examples - packed_examples_row: "
            #     + str(packed_examples_row))

            mask_row_cat_list = list([])
            for indexed_example_size in packed_examples_row:
                example_index = indexed_example_size.original_example_index
                example = examples[example_index]

            device = examples[0].get_device()
            mask_row_cat_list.append(self.create_row_mask_packed_mdlstm_computation(packed_examples_row, device))
            mask_extra_padding = self.create_mask_extra_padding(mask_row_cat_list, packed_examples_row)
            if mask_extra_padding is not None:
                mask_row_cat_list.append(mask_extra_padding)

            catted_mask_row = torch.cat(mask_row_cat_list, 1)

            # print("catted_row: " + str(catted_row))

            if len(mask_result_cat_list) > 0:
                mask_result_cat_list.append(self.create_vertical_mask_separator(example))

            mask_result_cat_list.append(catted_mask_row)

        mask_result = torch.cat(mask_result_cat_list, 0)

        return mask_result

    def create_vertically_and_horizontally_packed_examples_and_mask_one_direction(self, examples: list):

        result = self.create_vertically_and_horizontally_packed_examples_multiple_directions(examples, list([None]))
        mask_result = self.create_vertically_and_horizontally_packed_examples_mask_one_direction(examples)

        #
        # print("mask_result: " + str(mask_result))
        # print("result: " + str(result))

        # print("result.size(): " + str(result.size()))
        # packed_examples_2d = result.squeeze(1)
        # packed_examples_2d = packed_examples_2d.squeeze(0)
        # util.image_visualization.imshow_tensor_2d(mask_result.cpu())
        # util.image_visualization.imshow_tensor_2d(packed_examples_2d.cpu())

        # Sanity check to see that the result and the mask are of the same height and
        # width
        if (result.size(2) != mask_result.size(0)) or (result.size(3) != mask_result.size(1)):
            raise RuntimeError("Error: size of result " + str(result.size()) +
                               " and generated mask " + str(mask_result.size()) +
                               " are not compatible")

            # print("Percentage of real (non-padding) cells: " + str(100 * self.get_non_padding_fraction()) + "%")

        return result, mask_result

    @staticmethod
    def create_four_directions_tensor_flippings():
        no_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(False, False)
        # Flipping 2nd dimension
        height_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, False)
        # Flipping 3th dimension
        width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(False, True)
        # Flipping 2nd and 3th dimension combined
        height_and_width_flipping = util.tensor_flipping.TensorFlipping.create_tensor_flipping(True, True)

        result = list([no_flipping, height_flipping, width_flipping, height_and_width_flipping])
        return result

    def create_vertically_and_horizontally_packed_examples_four_directions_plus_mask(self, examples: list):

        tensor_flipping_list = MDLSTMExamplesPacking.create_four_directions_tensor_flippings()

        # # Original direction: left-to-right and top-to-bottom
        # packed_examples_direction_one = self.\
        #     create_vertically_and_horizontally_packed_examples_multiple_directions(examples, tensor_flipping_list[0])
        #
        # packed_examples_direction_two = self.\
        #     create_vertically_and_horizontally_packed_examples_multiple_directions(examples, tensor_flipping_list[1])
        #
        # packed_examples_direction_three = self. \
        #     create_vertically_and_horizontally_packed_examples_multiple_directions(examples, tensor_flipping_list[2])
        #
        # packed_examples_direction_four = self. \
        #     create_vertically_and_horizontally_packed_examples_multiple_directions(examples, tensor_flipping_list[3])
        #
        # # print("mdlstm_examples_packing - Created vertically and horizontally packed examples in four directions...")
        #
        # cat_list = list([packed_examples_direction_one, packed_examples_direction_two, packed_examples_direction_three,
        #                  packed_examples_direction_four])
        # # Concatenate the packed examples for different directions on the channels-dimension
        # result = torch.cat(cat_list, 1)

        result = self.create_vertically_and_horizontally_packed_examples_multiple_directions(examples,
                                                                                             tensor_flipping_list)

        mask_result = self.create_vertically_and_horizontally_packed_examples_mask_one_direction(examples)

        # print("mdlstm_examples_packing - Created mask...")

        #
        # print("mask_result: " + str(mask_result))
        # print("result: " + str(result))

        # print("result.size(): " + str(result.size()))
        # packed_examples_2d = result.squeeze(1)
        # packed_examples_2d = packed_examples_2d.squeeze(0)
        # util.image_visualization.imshow_tensor_2d(mask_result.cpu())
        # util.image_visualization.imshow_tensor_2d(packed_examples_2d.cpu())

        # Sanity check to see that the result and the mask are of the same height and
        # width
        if (result.size(2) != mask_result.size(0)) or (result.size(3) != mask_result.size(1)):
            raise RuntimeError("Error: size of result " + str(result.size()) +
                               " and generated mask " + str(mask_result.size()) +
                               " are not compatible")

        # print("Percentage of real (non-padding) cells: " + str(100 * self.get_non_padding_fraction()) + "%")

        return result, mask_result

    def extract_unskewed_activations_packed_examples_row(self, activations_as_tensor,
                                                         packed_examples_row: list,
                                                         first_row_index: int):

        # print("activations_as_tensor.size(): " + str(activations_as_tensor.size()))

        skewed_image_rows = packed_examples_row[0].example_size.height
        original_image_columns = self.get_packed_example_widths_plus_separator_overhead(packed_examples_row)

        # print("row_number: (original_image_columns + row_number: " +
        #       str(first_row_index) + ":" + str(original_image_columns + first_row_index))

        # The first row of this example row is at vertical index first_row_index (not 0) !!!
        activations_unskewed = activations_as_tensor[:, :, first_row_index, 0:original_image_columns]
        activations_unskewed = torch.unsqueeze(activations_unskewed, 2)

        for relative_row_number in range(1, skewed_image_rows):
            # print("row_number: (original_image_columns + row_number: " +
            #      str(relative_row_number) + ":" + str(original_image_columns + relative_row_number))

            absolute_row_number = first_row_index + relative_row_number
            # print("extract_unskewed_activations_packed_examples_row - absolute row number: "
            # + str(absolute_row_number))
            activation_columns = \
                activations_as_tensor[:, :, absolute_row_number,
                                      relative_row_number: (relative_row_number + original_image_columns)]
            activation_columns = torch.unsqueeze(activation_columns, 2)
            # print("activations_columns.size():" + str(activation_columns.size()))
            # print("activations_unskewed.size():" + str(activations_unskewed.size()))
            activations_unskewed = torch.cat((activations_unskewed, activation_columns), 2)

        # print("extract_unskewed_activations_packed_examples_row - activations_unskewed: "
        #       + str(activations_unskewed))

        activations_unskewed_split_list = list([])
        activations_unskewed_split_list.append(packed_examples_row[0].example_size.width)
        for indexed_example_size in packed_examples_row[1:len(packed_examples_row)]:
            activations_unskewed_split_list.append(self.example_separator_width)
            activations_unskewed_split_list.append(indexed_example_size.example_size.width)

        example_activations_list_with_padding = torch.split(activations_unskewed,
                                                            activations_unskewed_split_list,
                                                            3)
        example_activations_list = list([])

        # Extract the data and discard the padding
        for i in range(0, len(example_activations_list_with_padding), 2):
            example_activations_list.append(example_activations_list_with_padding[i])

        # print("extract_unskewed_activations_packed_examples_row - example_activations_list: " +
        #       str(example_activations_list))

        return example_activations_list

    def get_num_examples(self):
        return len(self.original_example_index_to_packed_index_table)

    def reorder_result_tensors_to_original_order(self, result_tensors_packed_order_list):
        result = list([])
        for original_example_index in range(0, self.get_num_examples()):
            packed_index = self.original_example_index_to_packed_index_table[original_example_index]
            result.append(result_tensors_packed_order_list[packed_index])
        return result

    def extract_unskewed_activations_from_activation_tensor(self, activations_as_tensor):

        result_tensors_packed_order = list([])

        first_row_index = 0
        for packed_examples_row in self.packed_examples:
            example_activations_list = self.\
                extract_unskewed_activations_packed_examples_row(activations_as_tensor,
                                                                 packed_examples_row,
                                                                 first_row_index)
            skewed_image_rows = packed_examples_row[0].example_size.height
            # Update first row index by height of packed_examples_row plus one
            # for vertical separator row
            first_row_index += skewed_image_rows + 1

            result_tensors_packed_order.extend(example_activations_list)

        # The result tensors are first in the order of the packing, we must
        # restore the original example order before returning the result
        return self.reorder_result_tensors_to_original_order(result_tensors_packed_order)

    def extract_unskewed_examples_activations_from_activation_columns(self, activation_columns):

        activations_as_tensor = ImageInputTransformer. \
            convert_activation_columns_list_to_tensor(activation_columns)
        # print("mdlstm_examples_packing - activations as tensor (including padding activations): "
        #       + str(activations_as_tensor))
        return self.extract_unskewed_activations_from_activation_tensor(activations_as_tensor)

    @staticmethod
    def extract_flipped_back_activations_from_unskewed_activations(activations_unskewed):
        tensor_flipping_list = MDLSTMExamplesPacking.create_four_directions_tensor_flippings()
        result = list([])
        for tensor in activations_unskewed:
            direction_tensors = torch.chunk(tensor, 4, 1)

            cat_list = list([])
            for i, direction_tensor in enumerate(direction_tensors):
                tensor_flipping = tensor_flipping_list[i]
                if tensor_flipping is not None:
                    cat_list.append(tensor_flipping.flip(direction_tensor))
                else:
                    cat_list.append(direction_tensor)
            # Concatenate the tensors along the channel dimension and add to the result
            result.append(torch.cat(cat_list, 1))

        return result


def test_create_vertically_and_horizontally_packed_examples():
    print("test_create_vertically_and_horizontally_packed_examples...")
    examples_list = list([])
    examples_list.append(torch.ones(1, 4, 16) * 1)
    examples_list.append(torch.ones(1, 4, 8) * 2)
    examples_list.append(torch.ones(1, 2, 2) * 3)
    examples_list.append(torch.ones(1, 2, 2) * 4)
    examples_list.append(torch.ones(1, 2, 12) * 5)
    examples_list = Utils.move_tensor_list_to_device(examples_list, 0)

    mdlstm_examples_packing = MDLSTMExamplesPacking.created_mdlstm_examples_packing(examples_list, 1)
    packed_examples, packed_examples_mask = mdlstm_examples_packing.\
        create_vertically_and_horizontally_packed_examples_and_mask_one_direction(examples_list)

    # Visualize the packed_examples and the packed_examples_mask to check
    # that they are as expected
    packed_examples_2d = packed_examples.squeeze(1)
    packed_examples_2d = packed_examples_2d.squeeze(0)
    print("Visualizing packed examples...")
    util.image_visualization.imshow_tensor_2d(packed_examples_2d.cpu())
    print("Visualizing packed examples mask...")
    util.image_visualization.imshow_tensor_2d(packed_examples_mask.cpu())


def test_pack_examples():
    print("Test pack examples...")
    examples_list = list([])
    examples_list.append(torch.zeros(1, 64, 528))
    examples_list.append(torch.zeros(1, 64, 128))
    examples_list.append(torch.zeros(1, 64, 128))
    examples_list.append(torch.zeros(1, 128, 256))
    examples_list.append(torch.zeros(1, 64, 256))
    examples_list.append(torch.zeros(1, 64, 64))
    examples_list.append(torch.zeros(1, 128, 256))
    examples_list.append(torch.zeros(1, 64, 384))

    mdlstm_examples_packing = MDLSTMExamplesPacking.created_mdlstm_examples_packing(examples_list, 8)
    # for packed_examples_row in mdlstm_examples_packing.packed_examples:
    #     print("packed_examples row: " + str(packed_examples_row))
    #     print("\n")
    mdlstm_examples_packing.print_packed_examples_rows()
    # print("Percentage of real (non-padding) pixels: " + str(100 * mdlstm_examples_packing.get_non_padding_fraction())
    #       + "%")


def test_pack_examples_of_same_height():
    print("Test pack examples of same height...")
    examples_list = list([])
    examples_list.append(IndexedExampleSize.create_indexed_example_size(0, 64, 528))
    examples_list.append(IndexedExampleSize.create_indexed_example_size(1, 64, 128))
    examples_list.append(IndexedExampleSize.create_indexed_example_size(2, 64, 128))
    examples_list.append(IndexedExampleSize.create_indexed_example_size(3, 64, 256))
    examples_list.append(IndexedExampleSize.create_indexed_example_size(4, 64, 64))
    examples_list.append(IndexedExampleSize.create_indexed_example_size(5, 64, 384))

    maximum_example_width = 528
    packed_examples = MDLSTMExamplesPacking.greedy_pack_examples_of_same_height(examples_list, maximum_example_width, 8)
    for packed_examples_row in packed_examples:
        print("packed_examples row: " + str(packed_examples_row))
        print("\n")


def main():
    #test_pack_examples_of_same_height()
    #test_pack_examples()
    test_create_vertically_and_horizontally_packed_examples()


if __name__ == "__main__":
    main()
