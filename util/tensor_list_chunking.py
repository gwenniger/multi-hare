import torch.tensor
from util.utils import Utils
from modules.size_two_dimensional import SizeTwoDimensional
from util.tensor_chunking import TensorChunking
import util.timing
from util.tensor_utils import TensorUtils
from collections import OrderedDict

# This class takes care of chunking a list of four-dimensional image tensors with
# list elements of the dimensions
# 1: channels, 2: height, 3: width
# into block tensors  with size 1: channels, 2: block_height, 3: block_width
# these blocks are concatenated along the batch_size dimension, to facilitate
# parallellization of computations that are run on the blocks. Finally, the class
# contains a method "dechunk_block_tensor_concatenated_along_batch_dimension"
# that restores the original list, 2-dimensional block configuration. This method requires
# the block dimension to remain the same, but allows the number of channels to change.
# This is to facilitate the typical use case, where an input tensor is chunked into
# blocks, features are computed for each of these blocks with an increased number of feature
# channels in the output, and finally the output blocks are rearranged to be in the original
# input configuration.
# This class is similar to TensorChunking, but the functionality is changed
# to the scenario where there is a list of 3d tensors of different sizes
# to be chunked into one tensor of blocks and then reconstructed to the original
# size, rather than a 4-dimensional tensor with the first dimension the batch
# dimension.


class TensorListChunking:

    def __init__(self, original_sizes: SizeTwoDimensional,
                 block_size: SizeTwoDimensional):
        self.original_sizes = original_sizes
        self.block_size = block_size
        TensorListChunking.check_block_size_fits_into_original_sizes(block_size, original_sizes)
        return

    @staticmethod
    def create_tensor_list_chunking(tensor_list: list,
                                    block_size: SizeTwoDimensional):
        return TensorListChunking(TensorListChunking.get_original_sizes_from_tensor_list(tensor_list), block_size)

    @staticmethod
    def get_original_sizes_from_tensor_list(tensor_list: list):
        result = list([])
        for x in tensor_list:
            if TensorUtils.number_of_dimensions(x) != 3:
                raise RuntimeError("Error: tenor x with size " + str(x.size()) +
                                   " does not have 3 dimensions, as required")

            # print("x.size(): " + str(x.size()))
            original_size = SizeTwoDimensional.create_size_two_dimensional(x.size(1), x.size(2))
            # print("original_size: " + str(original_size))
            result.append(original_size)
        # print(" get_original_sizes_from_tensor_list - result: " + str(result))
        return result

    @staticmethod
    def create_indices_list(number_of_examples: int, number_of_feature_blocks_per_example: int):
        result = []
        for i in range(0, number_of_examples):
            for j in range(0, number_of_feature_blocks_per_example):
                index = i + number_of_examples * j
                result.append(index)
        return result

    @staticmethod
    def create_torch_indices_selection_tensor(number_of_examples: int, number_of_feature_blocks_per_example: int):
        indices = TensorChunking.create_indices_list(number_of_examples, number_of_feature_blocks_per_example)
        result = torch.LongTensor(indices)
        return result

    def chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once(self,
                                                                                 tensor_list: list):

        # New implementation: collect everything and call torch.cat only once
        list_for_cat = list([])

        for tensor in tensor_list:

            # A dimension 0 must be added with unsqueeze,
            # for concatenation along the batch dimension later on
            tensor = tensor.unsqueeze(0)

            tensor_split_on_height = torch.split(tensor, self.block_size.height, 2)

            for row_block in tensor_split_on_height:
                blocks = torch.split(row_block, self.block_size.width, 3)
                # print("blocks: " + str(blocks))
                list_for_cat.extend(blocks)
        result = torch.cat(list_for_cat, 0)
        # print("chunk_tensor_into_blocks_concatenate_along_batch_dimension - result.size(): " + str(result.size()))

        return result

    def chunk_tensor_list_into_blocks_concatenate_along_batch_same_height_groups(self,
                                                                                 tensor_list: list):
        current_same_height_tensors_height = tensor_list[0].size(1)
        same_height_tensors = list([])

        cat_list = list([])

        for tensor in tensor_list:
            height = tensor.size(1)
            if height != current_same_height_tensors_height:
                tensor_blocks = self.\
                    chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once_fast_catlist(
                        same_height_tensors)
                cat_list.extend(tensor_blocks)
                same_height_tensors = list([])

            same_height_tensors.append(tensor)

        # Add last element
        same_height_tensor_blocks = self. \
            chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once_fast_catlist(
                same_height_tensors)
        cat_list.extend(same_height_tensor_blocks)

        return torch.cat(cat_list, 0)

    @staticmethod
    def group_examples_by_height(tensor_list):

        dictionary = OrderedDict([])
        for index, tensor in enumerate(tensor_list):
            height = tensor.size(1)
            if height in dictionary:
                same_height_list, original_indices_for_height_list = dictionary[height]
                same_height_list.append(tensor)
                original_indices_for_height_list.append(index)
            else:
                tensor_indices_tuple = list([tensor]), list([index])
                dictionary[height] = tensor_indices_tuple

        reordered_elements_list = list([])
        original_indices = list([])

        for height in dictionary.keys():
            same_height_list, original_indices_for_height_list = dictionary[height]
            reordered_elements_list.extend(same_height_list)
            original_indices.extend(original_indices_for_height_list)

        return reordered_elements_list, original_indices

    @staticmethod
    def retrieve_original_order(reordered_elements_list, original_indices):
        lookup_table = [0] * len(original_indices)
        for i, original_index in enumerate(original_indices):
            lookup_table[original_index] = i

        result = list([])
        for i in range(0, len(reordered_elements_list)):
            original_index = lookup_table[i]
            original_element = reordered_elements_list[original_index]
            result.append(original_element)
        return result


    def tensor_block_width(self, tensor):
        return int(tensor.size(2) / self.block_size.width)

    def tensor_block_height(self, tensor):
        return int(tensor.size(1) / self.block_size.height)

    """
    A block-list is effectively obtained by concatenating examples 
    in the width direction and then extracting blocks block-row by block-row.
    But this compromises the order of blocks in which all blocks for example 1 come
    before those for example 2, example 3 ...
    This method computes the order required to re-index the block list in the 
    wrong order, such as to restore the order of blocks example after example.  
    """
    def compute_block_re_indexing_order(self, tensor_list: list):
        cumulative_example_blockrow_widths = list([0])

        cumulative_sum_widths = 0
        for tensor in tensor_list:
            cumulative_sum_widths += self.tensor_block_width(tensor)
            # print("compute_block_re_indexing_order - cumulative_sum_widths: " + str(cumulative_sum_widths))
            cumulative_example_blockrow_widths.append(cumulative_sum_widths)

        device = tensor_list[0].get_device()
        cat_list = list([])
        number_of_block_rows = self.tensor_block_height(tensor_list[0])
        for example_index, example in enumerate(tensor_list):
            for block_row_index in range(0, number_of_block_rows):
                offset = cumulative_example_blockrow_widths[example_index]
                offset += cumulative_sum_widths * block_row_index
                # print("offset: " + str(offset))
                indices = torch.arange(offset, offset + self.tensor_block_width(example),
                                       dtype=torch.long,
                                       device=device)
                # print("indices : " + str(indices))
                cat_list.append(indices)
        # print("compute_block_re_indexing_order - result: " + str(result))
        result = torch.cat(cat_list, 0)
        return result

    @staticmethod
    def compute_permuted_tensor(tensor, indices_tensor):
        return torch.index_select(tensor, 0, indices_tensor)

    @staticmethod
    def compute_permuted_tensor_list(tensor_list, indices_tensor):
        result = list([])
        for i in range(0, indices_tensor.size(0)):
            original_index = indices_tensor[i]
            result.append(tensor_list[original_index])
        return result

    def chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once_fast_catlist(self, tensor_list: list):
        # New implementation: collect everything and call torch.cat only once

        # print("tensor_list[0].size(): " + str(tensor_list[0].size()))

        # This implementation saves out a for loop by concatenating all the tensors
        # along the width dimension first
        # This only works if the elements in the tensor_list all have the same
        # height and number of channels
        tensor_list_concatenated = torch.cat(tensor_list, 2)

        # A dimension 0 must be added with unsqueeze,
        # for concatenation along the batch dimension later on
        tensor_list_concatenated = tensor_list_concatenated.unsqueeze(0)
        tensor_split_on_height = torch.split(tensor_list_concatenated, self.block_size.height, 2)

        list_for_cat_wrong_order = list([])
        for row_block in tensor_split_on_height:
            # print("row_block: " + str(row_block))
            blocks = torch.split(row_block, self.block_size.width, 3)
            # print("blocks: " + str(blocks))
            list_for_cat_wrong_order.extend(blocks)

        order_restoring_permutation_tensor = self.compute_block_re_indexing_order(tensor_list)
        list_for_cat_right_order = TensorListChunking.compute_permuted_tensor_list(list_for_cat_wrong_order,
                                                                                   order_restoring_permutation_tensor)
        return list_for_cat_right_order

    def chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once_fast(self, tensor_list: list):

        list_for_cat_right_order = self.\
            chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once_fast_catlist(tensor_list)
        result = torch.cat(list_for_cat_right_order, 0)
        # result_wrong_order = torch.cat(list_for_cat_wrong_order, 0)
        # result = TensorListChunking.compute_permuted_tensor(result_wrong_order, order_restoring_permutation_tensor)
        # print("chunk_tensor_into_blocks_concatenate_along_batch_dimension - result_wrong_order.size(): " + str(result_wrong_order.size()))
        # print("chunk_tensor_into_blocks_concatenate_along_batch_dimension - result_wrong_order: " + str(result_wrong_order))

        return result

    # Chunks a list of three-dimensional tensors into blocks.
    # The first element dimension is the input channels,
    # the second and third dimension are the height and width respectively, along which
    # the chunking will be done. The result is formed by concatenating the blocks along
    # a new batch dimension
    def chunk_tensor_list_into_blocks_concatenate_along_batch_dimension(self,
                                                                        tensor_list: list,
                                                                        tensors_all_have_same_height: bool):
        # time_start = util.timing.date_time_start()

        if tensors_all_have_same_height:
            result = self.chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once_fast(tensor_list)
        else:
            # result = self.\
            #     chunk_tensor_list_into_blocks_concatenate_along_batch_dimension_cat_once(tensor_list)

            # New implementation, which tries to exploit groups of tensors of the same height
            # to perform the chunking faster
            result = self.\
                chunk_tensor_list_into_blocks_concatenate_along_batch_same_height_groups(tensor_list)

        # print("chunk_tensor_list_into_blocks_concatenate_along_batch_dimension - time used: \n" +
        #      str(util.timing.milliseconds_since(time_start)) + " milliseconds.")
        return result

        # No-cat implementation: slower on loss.backward
        # return self.chunk_tensor_into_blocks_concatenate_along_batch_dimension_no_cat(tensor)

    @staticmethod
    def check_block_size_fits_into_original_sizes(block_size: SizeTwoDimensional,
                                                  original_sizes: list):
        for original_size in original_sizes:
            TensorChunking.check_block_size_fits_into_original_size(block_size, original_size)

    # Reconstruct the tensor block row, using a combination of torch.split, squeeze and torch.cat operations
    # which for some reason yields better loss.backward performance than the original
    # implementation using torch.cat with a for loop and slicing
    # Note: this method is highly similar to the method with the same name in TensorChunking,
    # but works with 4D instead of 5D input tensors
    @staticmethod
    def reconstruct_tensor_block_row_split_squeeze_cat(row_slice):
        tensors_split = torch.split(row_slice, 1, 0)
        # print("tensors_split[0].size(): " + str(tensors_split[0].size()))
        # Splitting still retains the extra first dimension, which must be removed then
        # for all list elements using squeeze
        tensors_split_squeezed = list([])
        for element in tensors_split:
            tensors_split_squeezed.append(element.squeeze(0))
        # In "one go" with a combination of split, squeeze and cat
        return torch.cat(tensors_split_squeezed, 2)

    # Optimized version of "reconstruct_tensor_block_row_split_squeeze_cat"
    # without for loop, performing squeeze as last step
    # on the result tensor
    @staticmethod
    def reconstruct_tensor_block_row_split_cat_squeeze(row_slice):
        tensors_split = torch.split(row_slice, 1, 0)
        # Splitting still retains the extra first dimension,
        # which must be removed then after concatenation,
        # using squeeze(0)
        return torch.cat(tensors_split, 3).squeeze(0)

    # Reconstructs a tensor block row from a 4 dimensional tensor whose first dimension
    # goes over the blocks int the original tensor, and whose other three dimensions go over
    # the channel dimension and height and width dimensions of these blocks
    @staticmethod
    def reconstruct_tensor_block_row(row_slice):
        # return TensorListChunking.reconstruct_tensor_block_row_split_squeeze_cat(row_slice)
        # Further optimized version, avoiding for loop altogether, squeeze as last step
        return TensorListChunking.reconstruct_tensor_block_row_split_cat_squeeze(row_slice)

    def get_number_of_blocks_for_example(self, example_index):
        original_size = self.original_sizes[example_index]
        blocks_per_row = int(original_size.width / self.block_size.width)
        blocks_per_column = int(original_size.height / self.block_size.height)
        blocks_for_example = blocks_per_column * blocks_per_row
        return blocks_per_column, blocks_per_row, blocks_for_example

    def get_number_of_blocks_for_examples(self):
        blocks_per_row_list = list([])
        blocks_per_column_list = list([])
        blocks_for_examples_list = list([])
        for example_index in range(0, len(self.original_sizes)):
            blocks_per_column, blocks_per_row, blocks_for_example = \
                self.get_number_of_blocks_for_example(example_index)
            blocks_per_column_list.append(blocks_per_column)
            blocks_per_row_list.append(blocks_per_row)
            blocks_for_examples_list.append(blocks_for_example)
        return blocks_per_column_list, blocks_per_row_list, blocks_for_examples_list

    """
    Reconstruct tensor from a tensor grouped by blocks on the first dimension,
    by reconstructing the tensor rows row-by-row, and then concatenating 
    these rows
    """
    @staticmethod
    def reconstruct_tensor_row_by_row(tensor_grouped_by_block,
                                      blocks_per_column,
                                      blocks_per_row):

        row_slices = torch.split(tensor_grouped_by_block, blocks_per_row, 0)
        # print("tensor_grouped_by_block: " + str(tensor_grouped_by_block))

        tensor_block_row = TensorListChunking.reconstruct_tensor_block_row(row_slices[0])
        reconstructed_example_tensor = tensor_block_row

        for row_index in range(1, blocks_per_column):
            tensor_block_row = TensorListChunking.reconstruct_tensor_block_row(row_slices[row_index])
            reconstructed_example_tensor = torch.cat((reconstructed_example_tensor, tensor_block_row), 1)
        return reconstructed_example_tensor

    """
    Optimized reconstruction of a tensor grouped by blocks on the first dimension:
    1) Get a list of blocks
    2) Concatenate all the blocks into one long row, that is all the rows of the 
       to be reconstructed tensors are concatenated together
    3) Chunk the long single-row-of-blocks tensor again into the number of blocks
       per column in the result tensor, reconstructing the block rows of the final 
       result
    4) Finally concatenate these block rows to obtain the final result
    
    By "over-concatenating" the blocks in step 2, then chunking again in step 3, this 
    implementation avoids needing a for-loop to reconstruct the block-rows row by 
    row and then concatenating them. Because the torch.split / torch.cat methods 
    are likely to be better parallelized than for-loop-based implementations, this 
    is expected to be faster          
    """
    @staticmethod
    def reconstruct_tensor_cat_split_cat(tensor_grouped_by_block,
                                         blocks_per_column):
        block_tensors = torch.split(tensor_grouped_by_block, 1, 0)
        all_blocks_in_one_row = torch.cat(block_tensors, 3).squeeze(0)
        block_rows = torch.chunk(all_blocks_in_one_row, blocks_per_column, 2)
        reconstructed_example_tensor = torch.cat(block_rows, 1)

        return reconstructed_example_tensor

    """
        This function performs the inverse of 
        "chunk_tensor_into_blocks_concatenate_along_batch_dimension" : 
        it takes a tensor that is chunked into blocks, with the blocks stored along the
        first (batch) dimensions. It then reconstructs the original tensor from these blocks.
        The reconstruction is done using the "torch.cat" method, which preserves gradient
        information. Simply pasting over tensor slices in a newly created zeros tensor
        leads to a faulty implementation, as this does not preserve gradient information.
    """
    def dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size(self, tensor: torch.tensor,
                                                                                   block_size: SizeTwoDimensional):
        time_start = util.timing.date_time_start()

        number_of_examples = len(self.original_sizes)

        # print(">>> dechunk_block_tensor_concatenated_along_batch_dimension: - tensor.grad_fn "
        #      + str(tensor.grad_fn))

        # print("tensor.size(): " + str(tensor.size()))
        channels = tensor.size(1)

        result = list([])

        blocks_per_column_list, blocks_per_row_list, blocks_for_examples_list = \
            self.get_number_of_blocks_for_examples()

        # print("Total blocks for examples: " + str(sum(blocks_for_examples_list)))
        # print("tensor.size(): " + str(tensor.size()))
        example_sub_tensors = torch.split(tensor, blocks_for_examples_list, 0)

        blocks_start_index = 0
        for example_index in range(0, number_of_examples):
            blocks_per_column = blocks_per_column_list[example_index]
            # blocks_per_row = blocks_per_row_list[example_index]
            blocks_for_example = blocks_for_examples_list[example_index]
            example_sub_tensor = example_sub_tensors[example_index]
            # print("example_sub_tensor: " + str(example_sub_tensor))

            tensor_grouped_by_block = example_sub_tensor.view(blocks_for_example, channels,
                                                              block_size.height, block_size.width)
            # reconstructed_example_tensor = TensorListChunking.reconstruct_tensor_row_by_row(tensor_grouped_by_block,
            #                                                                                 blocks_per_column,
            #                                                                                 blocks_per_row)
            reconstructed_example_tensor = TensorListChunking.reconstruct_tensor_cat_split_cat(tensor_grouped_by_block,
                                                                                               blocks_per_column)
            result.append(reconstructed_example_tensor)

            # print(">>> dechunk_block_tensor_concatenated_along_batch_dimension: - result.grad_fn "
            #      + str(result.grad_fn))

            blocks_start_index += blocks_for_example

        # print("dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size - time used: \n" +
        #      str(util.timing.milliseconds_since(time_start)) + " milliseconds.")

        return result

    def dechunk_block_tensor_concatenated_along_batch_dimension(self, tensor: torch.tensor):
        return self.dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size(tensor, self.block_size)


def test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original(
        tensor_one, tensor_two, block_size, tensors_all_have_same_height: bool):

    if Utils.use_cuda():
        tensor_one = tensor_one.cuda()
        tensor_two = tensor_two.cuda()

    print("tensor_one: " + str(tensor_one))
    print("tensor_two: " + str(tensor_two))
    #print("tensor_one[0,  :, :]: " + str(tensor_one[0, :, :]))
    #print("tensor_two[0,  :, :]: " + str(tensor_two[0, :, :]))

    tensor_list = list([tensor_one, tensor_two])
    tensor_chunking = TensorListChunking.create_tensor_list_chunking(tensor_list, block_size)
    chunking = tensor_chunking.\
        chunk_tensor_list_into_blocks_concatenate_along_batch_dimension(tensor_list,
                                                                        tensors_all_have_same_height)
    print("chunking: " + str(chunking))
    print("chunking.size(): " + str(chunking.size()))
    dechunked_tensor_list = tensor_chunking.\
        dechunk_block_tensor_concatenated_along_batch_dimension_changed_block_size(chunking, block_size)

    print("dechunked_tensor_list: " + str(dechunked_tensor_list))

    # https://stackoverflow.com/questions/32996281/how-to-check-if-two-torch-tensors-or-matrices-are-equal
    # https://discuss.pytorch.org/t/tensor-math-logical-operations-any-and-all-functions/6624
    for tensor_original, tensor_reconstructed in zip(tensor_list, dechunked_tensor_list):
        tensors_are_equal =  torch.eq(tensor_original, tensor_reconstructed).all()
        print("tensors_are_equal: " + str(tensors_are_equal))
        if not tensors_are_equal:
            raise RuntimeError("Error: original tensor " + str(tensor_original) +
                               " and dechunked tensor " + str(tensor_reconstructed) +
                               " are not equal")
        else:
            print("Success: original tensor and dechunked(chunked(tensor)) are equal")


def test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original_single_block_row(
        tensors_all_have_same_height: bool):
    tensor_one = torch.Tensor([range(1, 49)]).view(2, 2, 12)
    tensor_two = torch.Tensor([range(49, 65)]).view(2, 2, 4)
    block_size = SizeTwoDimensional.create_size_two_dimensional(2, 2)
    test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original(tensor_one,
                                                                                 tensor_two,
                                                                                 block_size,
                                                                                 tensors_all_have_same_height)


def test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original_multiple_block_rows(
        tensors_all_have_same_height: bool):
    tensor_one = torch.Tensor([range(1, 33)]).view(2, 4, 4)
    tensor_two = torch.Tensor([range(33, 65)]).view(2, 4, 4)
    block_size = SizeTwoDimensional.create_size_two_dimensional(2, 2)
    test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original(tensor_one,
                                                                                 tensor_two,
                                                                                 block_size,
                                                                                 tensors_all_have_same_height)


def main():
    test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original_single_block_row(False)
    test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original_multiple_block_rows(False)
    test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original_single_block_row(True)
    test_tensor_list_block_chunking_followed_by_dechunking_reconstructs_original_multiple_block_rows(True)

    reordered_elements = list(["e", "c", "a", "b", "d"])
    original_indices = list([4, 2, 0, 1, 3])
    print(TensorListChunking.retrieve_original_order(reordered_elements, original_indices))

if __name__ == "__main__":
    main()