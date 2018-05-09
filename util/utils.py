import torch


class Utils:

    @staticmethod
    def use_cuda():
        return torch.cuda.is_available()