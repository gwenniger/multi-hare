import torch
from modules.size_two_dimensional import SizeTwoDimensional
import util.tensor_utils

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class TensorBlockStacking:

    @staticmethod
    def value_required_to_make_multiple_of(value: int, multiple_of: int):
        rest = value % multiple_of
        if rest > 0:
            return multiple_of - rest
        return 0

    @staticmethod
    def rescale_tensor_by_stacking_tensor_blocks(input_tensor: torch.Tensor,
                                                 block_size: SizeTwoDimensional, padding_value: int):
        """
        Takes a 2D tensor as input and creates a new tensor whereby blocks of the original tensor
        are stacked on a new zeroth channel dimension
        :param input_tensor:
        :param block_size:
        :param padding_value:
        :return: A 3D output tensor, with dimension 0 the channels dimension, equal to
        """


        input_tensor = input_tensor.cuda()
        # print("input_tensor: " + str(input_tensor))
        print("input_tensor.size(): " + str(input_tensor.size()))
        image_height = input_tensor.size(0)
        image_width = input_tensor.size(1)
        padding_top = TensorBlockStacking.value_required_to_make_multiple_of(image_height, block_size.height)
        padding_left = TensorBlockStacking.value_required_to_make_multiple_of(image_width, block_size.width)
        p2d = torch.nn.ConstantPad2d((padding_left, 0, padding_top, 0), padding_value)
        tensor_padded = p2d(input_tensor)
        print("tensor_padded.size(): " + str(tensor_padded.size()))
        block_rows = torch.split(tensor_padded, block_size.height, 0)
        block_rows_concatenated = torch.cat(block_rows, 1)
        block_list = torch.split(block_rows_concatenated, block_size.width, 1)
        blocks_stacked = torch.stack(block_list, 0)
        print("blocks_stacked.size(): " + str(blocks_stacked.size()))
        # blocks_stacked_one_dimensional = blocks_stacked.view(blocks_stacked.size(0)
        #                                                     * blocks_stacked.size(1) * blocks_stacked.size(2))
        result = blocks_stacked.view(int(tensor_padded.size(0) / block_size.height),
                                     int(tensor_padded.size(1) / block_size.width),
                                     block_size.height * block_size.width)
        result = result.transpose(2, 1)
        result = result.transpose(1, 0)

        print("rescale_tensor_by_stacking_tensor_blocks - result.size(): " + str(result.size()))
        # Move the result back to the cpu
        result = result.cpu()

        return result


def test_tensor_block_stacking():
    tensor = torch.Tensor([range(1, 17)]).view(4, 4)
    print("original tensor: " + str(tensor))
    result = TensorBlockStacking.rescale_tensor_by_stacking_tensor_blocks(
        tensor, SizeTwoDimensional.create_size_two_dimensional(2, 2), 1)
    print("result: " + str(result))
    expected_result = torch.Tensor([[[1.,   3.],
                                    [9.,  11.]],

                                    [[2.,   4.],
                                    [10.,  12.]],

                                    [[5.,   7.],
                                    [13.,  15.]],

                                    [[6.,   8.],
                                    [14.,  16.]]])
    if not util.tensor_utils.TensorUtils.tensors_are_equal(result, expected_result):
        raise RuntimeError("Error: expected the result to be equal to : " + str(expected_result) +
                           " but got: " + str(result))


def main():
    test_tensor_block_stacking()


if __name__ == "__main__":
    main()