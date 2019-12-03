import torch
import torch.nn as nn

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


def test_conv1d():
    input_states_size = 32
    output_states_size = 8
    number_of_groups = 2
    convolution = nn.Conv1d(input_states_size, output_states_size, 1,
                                         groups=number_of_groups)

    print("convolution.weight.size():" + str(convolution.weight.size()))
    print("convolution.bias.size():" + str(convolution.bias.size()))


def main():
    test_conv1d()


if __name__ == "__main__":
    main()
