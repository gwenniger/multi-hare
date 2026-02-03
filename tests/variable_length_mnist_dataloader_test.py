import torch
import modules.train_multi_dimensional_rnn_ctc
# Setting the sharing strategy to "file_system" is necessary to make the test succeed
torch.multiprocessing.set_sharing_strategy('file_system')
"""
See also these points about the "too many fds" problem
https://github.com/pytorch/pytorch/issues/165532

Along with what has already been shared, I'd add a couple more points.

"
Check your /dev/shm directory. Ensure there is no torch_*** objects lying around. If so remove them.
Try using torch.multiprocessing.set_sharing_strategy('file_system')
We are working on add thread based dataloading workers to get around issues like this. Thread based dataloading workers #161044. But cleaning up your dataset setup is probably the first step. Nested dataset setup (one dataset calls into another one) can also cause similar issue.
"

"""


"""
This test class tests the dataloader for the variable-length MNIST data. The motivation is that there are can be problems
with DataLoader when it is ran with multiple workers and without setting:
"torch.multiprocessing.set_sharing_strategy('file_system')"
These problems are due to too many file descriptors being created. This test serves to exclude
the DataLoader generation and looping over the IAM batch examples as the cause for these problems.
This test fails without  "torch.multiprocessing.set_sharing_strategy('file_system')"
It cashes then with 
...
  File "./lib/python3.14/site-packages/torch/utils/data/dataloader.py", line 427, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "multi-hare/lib/python3.14/site-packages/torch/utils/data/dataloader.py", line 1170, in __init__
    w.start()
...
    raise ValueError('too many fds')
ValueError: too many fds

This probably has to do with :
https://github.com/pytorch/pytorch/issues/65198
https://github.com/pytorch/pytorch/issues/11201
https://github.com/pytorch/pytorch/issues/973

"""


def main():
    batch_size = 2
    minimize_horizontal_padding = False
    train_loader, test_loader = modules.train_multi_dimensional_rnn_ctc.get_variable_length_mnist_dataloaders(
        batch_size, minimize_horizontal_padding, True)

    print("Completed creating dataloaders")
    for i, data in enumerate(train_loader, 0):
        print("train batch: " + str(i))
        continue

    for i, data in enumerate(test_loader, 0):
        print("test batch: " + str(i))
        continue


if __name__ == "__main__":
    main()
