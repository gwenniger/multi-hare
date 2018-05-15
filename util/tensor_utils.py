import torch


class TensorUtils:

    @staticmethod
    def tensors_are_equal(tensor_one, tensor_two):
        # https://stackoverflow.com/questions/32996281/how-to-check-if-two-torch-tensors-or-matrices-are-equal
        # https://discuss.pytorch.org/t/tensor-math-logical-operations-any-and-all-functions/6624
        return torch.eq(tensor_one, tensor_two).all()
