import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

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

def test_padding():
    input = Variable(torch.rand(1, 1, 4))  # 1D (N, C, L)
    input_2d = input.unsqueeze(2)  # add a fake height
    result1 = F.pad(input_2d, (2, 0, 0, 0)).view(1, 1, -1)  # left padding (and remove height)
    print("result1: " + str(result1))
    result2 = F.pad(input_2d, (0, 2, 0, 0)).view(1, 1, -1)  # right padding (and remove height)
    print("result2: " + str(result2))

    matrix = Variable(torch.FloatTensor([[1, 1], [2, 2]]))
    matrix = matrix.unsqueeze(0)
    matrix = matrix.unsqueeze(0)
    print("matrix: " + str(matrix))
    result3 = F.pad(matrix, (0, 0, 4, 0))
    print("result3: " + str(result3))

    matrix2 = Variable(torch.FloatTensor([[1, 1], [2, 2]]))
    print("matrix2 before: " + str(matrix2))
    matrix2[:, 0] = matrix2[:, 0] + matrix2[:, 0]
    print("matrix2 after: " + str(matrix2))



def main():
    #test_convolution_one()
    #test_convolution_with_padding()
    test_padding()


if __name__ == "__main__":
    main()

