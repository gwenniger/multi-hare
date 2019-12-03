import torch
from ._functions import Scatter, Gather
from ._functions_lists import ScatterList, GatherList
from collections import OrderedDict
import custom_data_parallel.comm_list

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"
"""
Extended from the implementation at
https://github.com/pytorch/pytorch/tree/master/torch/nn/parallel
"""

def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            # print("scatter_gather - len(tuple) : " + str(len(obj)))
            # for element in obj:
            #     print("scatter_gather - tuple element: " + str(element))
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            # print("scatter_gather - len(obj) : " + str(len(obj)))
            # return list(map(list, zip(*map(scatter_map, obj))))
            # result = ScatterList.apply(target_gpus, None, obj)
            result = custom_data_parallel.comm_list.scatter_list(obj, target_gpus)
            # print("scatter_gather - scatter_map len(result): " + str(len(result)))
            return result
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def is_list_of_lists(out):
    if isinstance(out, list):
        if len(out) > 0 and isinstance(out[0], list):
            return True
    return False


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        # out = outputs[0]  # original code (normal scatter_gather)

        # When working wit lists this is needed
        out = outputs

        if isinstance(out, torch.Tensor):
            print("Scatter-Gather, gathering tensor...")
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))

        # https://discuss.pytorch.org/t/training-network-with-multiple-outputs-with-multi-gpus/6344/4
        if isinstance(out, OrderedDict):
            return OrderedDict(
                [(k, Gather.apply(target_device, dim, *[each[k] for each in outputs])) for k in out.keys()])

        # https://discuss.pytorch.org/t/training-network-with-multiple-outputs-with-multi-gpus/6344/4

        if is_list_of_lists(out):
            print("Scatter-Gather, gathering list...")
            print("scatter_gather - gather_map  - len(out): " + str(len(out)))
            # result = list(
            #     [(k, Gather.apply(target_device, dim, *[each[k] for each in outputs])) for k in out])
            # #return GatherList.apply(target_device, dim, *outputs)
            result = custom_data_parallel.comm_list.gather_list(out)

            # print("scatter_gather - gather_map  - len(result): " + str(len(result)))
            return result
        elif isinstance(out, list):
            # Gather a list of tensors as opposed to a list of lists of tensors
            if isinstance((out[0]), torch.Tensor):
                return Gather.apply(target_device, dim, *out)

        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        # print("gather - len(outputs): " + str(len(outputs)))
        return gather_map(outputs)
    finally:
        gather_map = None
