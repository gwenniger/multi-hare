import torch
import torch.nn as nn
import torch.autograd as autograd


class ConvolutationTests:

    def __init__(self):
        print("do nothing")


def test_convolution_one():
    input_channels = 1
    output_channels = 1
    kernel_size = 2
    # See: http://pytorch.org/docs/0.3.1/nn.html?highlight=conv1d#torch.nn.Conv1d
    m = nn.Conv1d(input_channels, output_channels, kernel_size, stride=1)
    tensor_data = [[[0, 0, 0, 1, 1, 1, 2, 2, 2]]]
    tensor = torch.FloatTensor(tensor_data)
    # See: http://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html
    input_values = autograd.Variable(tensor)
    # input_values = autograd.Variable(torch.zeros(1, input_channels, 3))
    # new_value = autograd.Variable(torch.ones(1))
    print("input: " + str(input_values))
    print("input.data: " + str(input_values.data))
    output = m(input_values)
    print("output: " + str(output))


def test_convolution_with_padding():
    input_channels = 1
    output_channels = 1
    kernel_size = 2
    # See: http://pytorch.org/docs/0.3.1/nn.html?highlight=conv1d#torch.nn.Conv1d
    m = nn.Conv1d(input_channels, output_channels, kernel_size, stride=1, padding=1)
    tensor_data = [[[0, 0, 0, 1, 1, 1, 2, 2, 2]]]
    tensor = torch.FloatTensor(tensor_data)
    # See: http://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html
    input_values = autograd.Variable(tensor)
    # input_values = autograd.Variable(torch.zeros(1, input_channels, 3))
    # new_value = autograd.Variable(torch.ones(1))
    print("input: " + str(input_values))
    print("input.data: " + str(input_values.data))
    output = m(input_values)
    print("output: " + str(output))



def main():
    #test_convolution_one()
    test_convolution_with_padding()


if __name__ == "__main__":
    main()

