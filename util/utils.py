import torch


class Utils:

    @staticmethod
    def use_cuda():
        return torch.cuda.is_available()

    @staticmethod
    def move_tensor_list_to_device(tensor_list, device):
        # We cannot copy an entire list to gpu using to(device)
        # so we need to do it one by one
        tensor_list_on_device = list([])
        for element in tensor_list:
            element = element.to(device)
            tensor_list_on_device.append(element)
        return tensor_list_on_device

