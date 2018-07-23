import torch
from torch.cuda import nccl
from torch._utils import _accumulate, _take_tensors, _flatten_dense_tensors, \
    _flatten_sparse_tensors, _unflatten_dense_tensors, \
    _unflatten_sparse_tensors, _reorder_tensors_as
from util.utils import Utils


def chunk_list(list_of_tensors: list, number_of_chunks: int):
    minimal_chunk_size = int(len(list_of_tensors) / number_of_chunks)
    elements_remaining = len(list_of_tensors) -(minimal_chunk_size * number_of_chunks)

    result = list([])
    start_index = 0
    for chunk_index in range(0, number_of_chunks):
        chunk_size_current_chunk = minimal_chunk_size
        if elements_remaining > 0:
            # The chunk size of the first chunks is bigger, to finish up
            # the elements remaining
            chunk_size_current_chunk += 1

        result.append(list_of_tensors[start_index:start_index+chunk_size_current_chunk])
        elements_remaining -= 1

    return result


def scatter_list(list_of_tensors, devices, chunk_sizes=None, streams=None):
    """Scatters tensor across multiple GPUs.

    Arguments:
        list_of_tensors (list(Tensor)): list of tensors to scatter.
        devices (Iterable[int]): iterable of ints, specifying among which
            devices the tensor should be scattered.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
            each device. It should match ``devices`` in length and sum to
            ``len(list_of_tensors)``. If not specified, the
            list of tensors will be divided
            into equal chunks.

    Returns:
        A tuple containing chunks of the ``list of tensors``, spread across given
        ``devices``.
    """
    if chunk_sizes is None:
        chunks = chunk_list(list_of_tensors, len(devices))
    else:
        assert sum(chunk_sizes) == len(list_of_tensors), "given chunk sizes " \
            "don't sum up to the tensor's size (sum(chunk_sizes) == {}, but " \
            "expected {})".format(sum(chunk_sizes), len(list_of_tensors))
        assert min(chunk_sizes) > 0, "got a negative chunk_size"
        # chunks = [list_of_tensors.narrow(dim, start - size, size)
        #           for start, size in zip(_accumulate(chunk_sizes), chunk_sizes)]
        chunks = [list_of_tensors[start:start+size]
                  for start, size in zip(_accumulate(chunk_sizes), chunk_sizes)]
    # chunks = tuple(chunk.contiguous() for chunk in chunks)
    # TODO: copy to a pinned buffer first (if copying from CPU)
    if streams is None:
        streams = [None] * len(devices)
    outputs = []
    for device, chunk, stream in zip(devices, chunks, streams):
        with torch.cuda.device(device), torch.cuda.stream(stream):
            # outputs.append(chunk.cuda(device, non_blocking=True))
            outputs.append(Utils.move_tensor_list_to_device(chunk, device, non_blocking=True))
    return tuple(outputs)


def gather_list(tensor_lists, destination=None):
    """Gathers tensor lists from multiple GPUs.

    Tensor sizes in all dimension different than ``dim`` have to match.

    Arguments:
        tensor_lists (Iterable[Tensor]): iterable of tensor lists to gather.
        destination (int, optional): output device (-1 means CPU, default:
            current device)

    Returns:
        A tensor list located on ``destination`` device, that is a result of
        appending``tensor lists``.
    """
    total_size = 0
    # expected_size = list(tensor_lists[0].size())
    for tensor_list in tensor_lists:
        for element in tensor_list:
            assert element.is_cuda, "gather expects all inputs to be on GPUs"
        # expected_size[dim] = tensor_list.size(dim)
        # if list(tensor_list.size()) != expected_size:
        #     got = 'x'.join(str(x) for x in tensor_list.size())
        #     expected = 'x'.join(str(x) for x in expected_size)
        #     raise ValueError("gather got an input of invalid size: got {}, "
        #                      "but expected {}".format(got, expected))
        #total_size += tensor_list.size(dim)
        total_size += len(tensor_list)
    # expected_size[dim] = total_size
    # expected_size = torch.Size(expected_size)

    result = list([])
    for tensor_list in tensor_lists:
        result.extend(tensor_list)

    if destination is None:
        destination = torch.cuda.current_device()
    if destination == -1:
        # result = tensor_lists[0].new().cpu().resize_(expected_size)
        result = Utils.move_tensor_list_to_device(result, -1)
    else:
        result = Utils.move_tensor_list_to_device(result, destination)

    # chunk_start = 0
    # # TODO: if copying to CPU, allocate a pinned buffer, do async copies to it,
    # # and copy it to regular memory
    # for tensor_list in tensor_lists:
    #     result.narrow(dim, chunk_start, tensor_list.size(dim)).copy_(tensor_list, True)
    #     chunk_start += tensor_list.size(dim)
    return result
