import torch


class Utils:

    @staticmethod
    def use_cuda():
        return torch.cuda.is_available()

    @staticmethod
    def move_tensor_list_to_device(tensor_list, device, non_blocking=True):
        # We cannot copy an entire list to gpu using to(device)
        # so we need to do it one by one
        tensor_list_on_device = list([])
        for element in tensor_list:
            if (isinstance(device, torch.device) and device.type == "cuda") or \
                    isinstance(device, int) and device >= 0:
                # print("move_tensor_list_to_device with \"cuda\" - device: " + str(device))
                # if isinstance(device, torch.device):
                    # print("move_tensor_list_to_device with \"cuda\" - device.type: " + str(device.type))
                # non-blocking option is only supported by "cuda" method, not by "to" method

                element = element.cuda(device, non_blocking=non_blocking)
            else:
                # print("move_tensor_list_to_device with \"to\": " + str(device))
                element = element.to(device)
            tensor_list_on_device.append(element)
        return tensor_list_on_device

