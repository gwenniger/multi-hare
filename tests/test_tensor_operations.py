import torch


def test_view():
    tensor_one = torch.Tensor([range(1, 49)]).view(2, 2, 2, 6)
    print("tensor_one\n" + str(tensor_one))
    print("tensor_one.size(): " + str(tensor_one.size()))
    tensor_one_no_height = tensor_one.view(tensor_one.size(0), tensor_one.size(1),
                                           tensor_one.size(2) * tensor_one.size(3))
    print("tensor_one_no_height:\n" + str(tensor_one_no_height))
    print("tensor_one_no_height.size(): " + str(tensor_one_no_height.size()))


def main():
    test_view()


if __name__ == "__main__":
    main()