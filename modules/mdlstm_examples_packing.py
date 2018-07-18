from modules.size_two_dimensional import SizeTwoDimensional
from sortedcontainers.sortedlist import SortedList
# See: https://stackoverflow.com/questions/31493603/
# alternative-to-pythons-sort-for-inserting-into-a-large-list-and-keeping-it
# http://www.grantjenks.com/docs/sortedcontainers/
# Sorted containers will be used to efficiently find the closest elements that
# that are smaller than some value
import torch


class IndexedExampleSize:

    def __init__(self,  original_example_index: int, example_size: SizeTwoDimensional):
        self.original_example_index = original_example_index
        self.example_size = example_size

    @staticmethod
    def create_indexed_example_size(original_example_index: int, height: int, width: int):
        size = SizeTwoDimensional.create_size_two_dimensional(height, width)

        return IndexedExampleSize(original_example_index, size)

    # We need to implement the comparable operators in order to be able
    # to make use of sortedcontainers
    # http://gerg.ca/blog/post/2012/python-comparison/
    def __eq__(self, other):
        return self.example_size == other.example_size and \
               self.original_example_index == other.original_example_index

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        if self.example_size.width > other.example_size.width:
            return True
        elif self.example_size.height > other.example_size.height:
            return True
        return False

    def __lt__(self, other):
        if self.example_size.width < other.example_size.width:
            return True
        elif self.example_size.height < other.example_size.height:
            return True
        return False

    def __ge__(self, other):
        return (self > other) or (self == other)

    def __le__(self, other):
        return (self < other) or (self == other)

   #http://gerg.ca/blog/post/2012/python-comparison/

    def __str__(self):
        result = "<I.E.Size> "
        result += "original_example_index: " + str(self.original_example_index) + " "
        result += "example_size: " + str(self.example_size) + " "
        result += "</I.E.Size>"
        return result

    def __repr__(self):
        return self.__str__()




class MDLSTMExamplesPacking:

    def __init__(self, packed_examples, max_example_width: int):
        self.packed_examples = packed_examples
        self.max_example_width = max_example_width

    @staticmethod
    def created_mdlstm_examples_packing(examples_list: list, example_separator_width: int):
        packed_examples, max_example_width = MDLSTMExamplesPacking.get_packed_examples(examples_list, example_separator_width)
        return MDLSTMExamplesPacking(packed_examples, max_example_width)

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
    def get_summed_example_widths(packed_examples_row: list):
        result = 0
        for indexed_example_size in packed_examples_row:
            result += indexed_example_size.example_size.width
        return result

    @staticmethod
    def get_packed_examples_row_height(packed_examples_row: list):
        return packed_examples_row[0].example_size.height

    @staticmethod
    def print_packed_examples_row(packed_examples_row: list):
        summed_example_widths = MDLSTMExamplesPacking.get_summed_example_widths(packed_examples_row)
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

        print("sorted_list: " + str(sorted_list))

        examples_height = examples_list[0].example_size.height

        result = list([])
        result_row = list([])
        space_remaining_current_row = maximum_example_width

        while len(sorted_list) > 0:
            print("len(sorted_list): " + str(len(sorted_list)))
            print("sorted_list: " + str(sorted_list))
            print("space_remaining_current_row: " + str(space_remaining_current_row))
            query_example = IndexedExampleSize.create_indexed_example_size(-1, examples_height,
                                                                           space_remaining_current_row)
            largest_element_smaller_than_max_width_index = sorted_list.bisect_right(query_example) - 1

            print("largest_element_smaller_than_max_width_index: " + str(largest_element_smaller_than_max_width_index))

            # No suitable smaller example is left
            if largest_element_smaller_than_max_width_index < 0:
                result.append(result_row)
                result_row = list([])
                space_remaining_current_row = maximum_example_width
                print("Starting new result_row...")
            else:
                largest_element_smaller_than_max_width = sorted_list.pop(largest_element_smaller_than_max_width_index)
                result_row.append(largest_element_smaller_than_max_width)
                space_remaining_current_row -= largest_element_smaller_than_max_width.example_size.width
                space_remaining_current_row -= example_separator_width

        # Add the last result row
        result.append(result_row)

        return result

    @staticmethod
    def get_maximum_example_width(example_sizes_list: list):
        result = 0
        for indexed_example_size in example_sizes_list:
            result = max(result, indexed_example_size.example_size.width)
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
            summed_widths = MDLSTMExamplesPacking.get_summed_example_widths(packed_examples_row)
            row_non_padding_pixels = height * summed_widths
            total_row_pixels = height * self.max_example_width

            total_non_padding_pixels += row_non_padding_pixels
            total_pixels += total_row_pixels

        return float(total_non_padding_pixels) / total_pixels

    def print_packed_examples_rows(self):
        for packed_examples_row in self.packed_examples:
            MDLSTMExamplesPacking.print_packed_examples_row(packed_examples_row)


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
    print("Percentage of real (non-padding) pixels: " + str(100 * mdlstm_examples_packing.get_non_padding_fraction())
          + "%")


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
    test_pack_examples_of_same_height()
    test_pack_examples()


if __name__ == "__main__":
    main()
