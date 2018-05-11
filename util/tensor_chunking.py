import torch.tensor
from util.utils import Utils
from modules.size_two_dimensional import SizeTwoDimensional


class TensorChunking:

    def __init__(self, batch_size:int, original_size: SizeTwoDimensional,
                 block_size: SizeTwoDimensional):
        self.batch_size = batch_size
        self.original_size = original_size
        self.block_size = block_size
        TensorChunking.check_block_size_fits_into_original_size(block_size, original_size)
        self.blocks_per_row = int(original_size.width / block_size.width)
        self.blocks_per_column = int(original_size.height / block_size.height)
        self.number_of_feature_blocks_per_example = self.blocks_per_column * self.blocks_per_row
        self.selection_tensor = TensorChunking.\
            create_torch_indices_selection_tensor(self.batch_size, self.number_of_feature_blocks_per_example)
        return

    @staticmethod
    def create_tensor_chunking(batch_size:int, original_size: SizeTwoDimensional,
                               block_size: SizeTwoDimensional):
        return TensorChunking(batch_size, original_size, block_size)

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


    # Chunks a four-dimensional tensor into blocks.
    # The fist dimension is the batch dimension, the second dimension the input channels,
    # the third and forth dimension are the height and width respectively, along which
    # the chunking will be done. The result is returned as a list
    @staticmethod
    def chunk_tensor_into_blocks_return_as_list(
            tensor: torch.tensor, block_size: SizeTwoDimensional):
        result = list([])
        tensor_split_on_height = torch.split(tensor, block_size.height, 2)
        for row_block in tensor_split_on_height:
            blocks = torch.split(row_block, block_size.width, 3)
            result.extend(blocks)
        return result


    # Chunks a four-dimensional tensor into blocks.
    # The fist dimension is the batch dimension, the second dimension the input channels,
    # the third and forth dimension are the height and width respectively, along which
    # the chunking will be done. The result is formed by concatenating the blocks along
    # the firs (batch) dimension
    def chunk_tensor_into_blocks_concatenate_along_batch_dimension(self,
            tensor: torch.tensor):
        result = torch.zeros(0, tensor.size(1), self.block_size.height,
                             self.block_size.width)
        tensor_split_on_height = torch.split(tensor, self.block_size.height, 2)
        for row_block in tensor_split_on_height:
            blocks = torch.split(row_block, self.block_size.width, 3)
            list_for_cat = list([])
            list_for_cat.append(result)
            list_for_cat.extend(blocks)
            result = torch.cat(list_for_cat, 0)
            print("result.size(): " + str(result.size()))
        return result

    @staticmethod
    def check_block_size_fits_into_original_size(block_size: SizeTwoDimensional,
            original_size: SizeTwoDimensional):
        # Check that things fit
        if (original_size.width % block_size.width) != 0:
            raise RuntimeError("Error: the original size is not a multiple of the"
                               "block size in the width dimension")

        if (original_size.height % block_size.height) != 0:
            raise RuntimeError("Error: the original size is not a multiple of the"
                               "block size in the height dimension")

    # This function performs the inverse of
    # "chunk_tensor_into_blocks_concatenate_along_batch_dimension" : it takes
    # a tensor that is chunked into blocks, with the blocks stored along the
    # first (batch) dimensions. It then reconstructs the original tensor from these blocks
    def dechunk_block_tensor_concatenated_along_batch_dimension(self, tensor: torch.tensor):
        number_of_examples = int(tensor.size(0) / self.number_of_feature_blocks_per_example)

        # if number_of_examples == self.batch_size:
        #     selection_tensor = self.selection_tensor
        # else:
        #     selection_tensor = TensorChunking.create_torch_indices_selection_tensor(
        #         number_of_examples, self.number_of_feature_blocks_per_example)
        #     if Utils.use_cuda():
        #         selection_tensor = selection_tensor.cuda()
        # # The blocks in tensor are ordered by example, that is first all blocks
        # # for example 1, then all blocks for example 2 etc.
        # # Reorder them to have first all blocks at position 1 (for all examples),
        # # then all blocks for position 2 etc.
        # blocks_reordered_grouped_by_block_position = \
        #     torch.index_select(tensor, 0, selection_tensor)
        #
        # print("blocks_reordered_grouped_by_block_position: " + str(blocks_reordered_grouped_by_block_position))
        print("tensor.size(): " + str(tensor.size()))
        channels = tensor.size(1)
        result = torch.zeros(number_of_examples, channels, self.original_size.height,
                             self.original_size.width)
        for i in range (0, tensor.size(0)):
            raise RuntimeError("not yet implemented")
            # TODO: Fix me


        # blocks_per_example
        return


def test_tensor_block_chunking():
    a = torch.Tensor([range(1, 97)]).view(2, 2, 4, 6)
    print(a)
    print("a[0, 0, :, :]: " + str(a[0, 0, :, :]))
    # chunking = chunk_tensor_into_blocks_return_as_list(
    #     a, SizeTwoDimensional.create_size_two_dimensional(2, 2))
    # print("chunking: " + str(chunking))
    # for item in chunking:
    #     print("item.size(): " + str(item.size()))
    batch_size = 2
    original_size = SizeTwoDimensional.create_size_two_dimensional(4, 6)
    block_size = SizeTwoDimensional.create_size_two_dimensional(2, 2)
    tensor_chunking = TensorChunking.create_tensor_chunking(batch_size, original_size, block_size)
    chunking = tensor_chunking.chunk_tensor_into_blocks_concatenate_along_batch_dimension(a)
    print("chunking: " + str(chunking))
    print("chunking.size(): " + str(chunking.size()))
    tensor_chunking.dechunk_block_tensor_concatenated_along_batch_dimension(chunking)


def main():
    test_tensor_block_chunking()


if __name__ == "__main__":
    main()