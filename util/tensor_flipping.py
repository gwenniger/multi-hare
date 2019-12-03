import torch
from  util.tensor_utils import TensorUtils

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"

# Code adapted from
# https://github.com/pytorch/pytorch/issues/229


# The TensorFlipping class implements flipping of tensors along the height and
# width dimensions, which are assumed to be the last two dimensions of the tensor.
# This is useful for implementing Multi-Directional MultiDimensionalLSTM
# Note that flipping is not the same as full rotation. As a special case, 180
# degree rotation is the same as flipping both the height and width dimensions
# Rotations of 90 degrees and 270 degrees however require not only one of the
# dimensions to be flipped, but also the x and y axis to be switched.
# For multi-directional LSTM implementation however, full rotation is not nescessary,
# flipping of axis is enough. Full rotation also does not yield real computational
# benefits, since the width of the skewed input matrix used for MultiDimensionalLSTM
# implementation is of width (original_height + original_width -1). In fact, it is
# slightly advantageous to have the original height dimension to be the shorter
# of original width and original height.
# For handwriting recognition of line strips this is typically already the case anyhow.
# But the skewed width is always the same, and
# since on GPU computation this is the dimension along which no parallelization is
# possible, that dimension is the most decisive for performance.
class TensorFlipping:

    def __init__(self, flip_height: bool, flip_width: bool):
        self.flip_height = flip_height
        self.flip_width = flip_width

    @staticmethod
    def create_tensor_flipping(flip_height: bool, flip_width: bool):
        return TensorFlipping(flip_height, flip_width)

    def flip(self, tensor):
        number_of_dimensions = len(tensor.size())
        height_dimension = number_of_dimensions - 2
        width_dimension = number_of_dimensions - 1
        # print("number_of_dimensions:" + str(number_of_dimensions))

        if self.flip_height:
            if self.flip_width:
                result = flip(flip(tensor, width_dimension), height_dimension)
            else:
                result = flip(tensor, height_dimension)
        elif self.flip_width:
            result = flip(tensor, width_dimension)
        else:
            result = tensor

        return result


# Flips the elements of a tensor along a dimension
def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim

    indices = tuple(slice(None, None) if i != dim
                    else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                    for i in range(x.dim()))
    return x[indices]


def test_tensor_flipping():
    # Code to test it with cpu
    a = torch.Tensor([range(1, 25)]).view(1, 2, 3, 4)
    print(a)
    print(flip(a, 0)) # Or -4
    print(flip(a, 1)) # Or -3
    print(flip(a, 2)) # Or -2
    print(flip(a, 3)) # Or -1

    print("######### CUDA #########")

    # Code to test it with cuda
    a = torch.Tensor([range(1, 25)]).view(1, 2, 3, 4).cuda()
    print(a)
    print(flip(a, 0)) # Or -4
    print(flip(a, 1)) # Or -3
    print(flip(a, 2)) # Or -2
    print(flip(a, 3)) # Or -1

    print("Combined flips...")
    print(flip(flip(a, 2), 3))

    ######
    x = torch.randn(3, 4).cuda()
    indices = torch.LongTensor([0, 2]).cuda()
    selection = torch.index_select(x, 0, indices).cuda()
    print("selection: " + str(selection))


def test_tensor_flipping_twice_retrieves_original():
    # a = torch.Tensor([range(1, 25)]).view(1, 2, 3, 4)
    a = torch.Tensor([range(1, 10)]).view(3, 3)
    print("a: " + str(a))

    flipping_tuples = list([])
    flipping_tuples.append((True, False))
    flipping_tuples.append((True, True))
    flipping_tuples.append((False, True))

    for flipping_tuple in flipping_tuples:
        print(">>> flip height: " + str(flipping_tuple[0]) + ", flip width: " + str(flipping_tuple[1]))
        tensor_flipping = TensorFlipping.create_tensor_flipping(flipping_tuple[0],
                                                                flipping_tuple[1])
        a_flipped = tensor_flipping.flip(a)
        print("a_flipped: " + str(a_flipped))
        a_flipped_back = tensor_flipping.flip(a_flipped)
        print("a_flipped_back: " + str(a_flipped_back))
        # a_flipped_back = torch.zeros(3, 3)

        if not TensorUtils.tensors_are_equal(a, a_flipped_back):
            raise RuntimeError("Error: original tensor:\n " +
                               str(a) + " and flipped, then flipped back tensor:\n " +
                               str(a_flipped_back) + " are not equal")


def main():
    test_tensor_flipping()
    test_tensor_flipping_twice_retrieves_original()


if __name__ == "__main__":
    main()
