import torch

# Code adapted from
# https://github.com/pytorch/pytorch/issues/229


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

def main():
    # test_mdrnn_cell()
    #test_mdrnn()
    test_tensor_flipping()


if __name__ == "__main__":
    main()
