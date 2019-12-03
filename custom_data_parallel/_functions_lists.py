import torch
import custom_data_parallel.comm_list as comm_list
from torch.autograd import Function

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"
"""
Extended from the implementation at
https://github.com/pytorch/pytorch/tree/master/torch/nn/parallel
"""

"""
These list variants of Scatter and Gather are eventually not used in the 
customized scatter_gather. 
The comm_list functions are used directly.
"""


class GatherList(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        print("Entered GatherList.forward...")
        assert all(map(lambda i: i.is_cuda, inputs))
        ctx.target_device = target_device
        ctx.dim = dim
        # ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs)) # original
        # Get the device for the first list element
        ctx.input_gpus = tuple(map(lambda i: i[0].get_device(), inputs))
        # ctx.input_sizes = tuple(map(lambda i: i.size(ctx.dim), inputs)) # original
        # Use the list length instead of the dimensionality
        ctx.input_sizes = tuple(map(lambda i: len(i), inputs))
        return comm_list.gather_list(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        print("Entered GatherList.backward...")
        return (None, None) + ScatterList.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)


class ScatterList(Function):

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, input):

        if not isinstance(input, list):
            raise RuntimeError("Error: input is not of type list")

        print("Entered ScatterList.forward...")

        ctx.target_gpus = target_gpus
        ctx.chunk_sizes = chunk_sizes
        # ctx.dim = dim
        ctx.input_device = input[0].get_device() if input[0].is_cuda else -1
        streams = None
        if ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in ctx.target_gpus]
        outputs = comm_list.scatter_list(input, ctx.target_gpus, ctx.chunk_sizes, streams)
        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(ctx.target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)

        print("ScatterList.forward - len(outputs): " + str(len(outputs)))
        # print("ScatterList.forward - outputs: " + str(outputs))
        print("Exiting ScatterList.forward...")

        # outputs_as_list_of_tuples = list([])
        # for element in outputs:
        #     outputs_as_list_of_tuples.append(tuple(element))

        #return outputs_as_list_of_tuples
        #return list([torch.zeros(4, 2), torch.zeros(4, 2)]) # does not work
        #return tuple([torch.zeros(4, 2), torch.zeros(4, 2)]) # does work

        # Struggling with this error:
        # TypeError: ScatterListBackward.forward: expected Variable (got tuple) for return value 0
        # Apparently it wants a list of variables to be returned
        # new_result = tuple(outputs_as_list_of_tuples) # doe not work
        # return new_result
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        print("Entered ScatterList.backward...")
        return None, None, None, GatherList.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams = None


def _get_stream(device):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * torch.cuda.device_count()
    if _streams[device] is None:
        _streams[device] = torch.cuda.Stream(device)
    return _streams[device]
