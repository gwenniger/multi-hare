""" This class will implement Multi-dimensional RNN modules and also Multi-dimensional LSTM modules
    These will be based on a generalization of the native pytorch RNN code
    from:  https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
"""

import math
import torch
import warnings
import itertools
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
import torch.nn.functional as F


class MDRNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, h1, h2,  hidden_label=''):
        #if input.size(0) != hx.size(0):
         #   raise RuntimeError(
         #       "Input batch size {} doesn't match hidden{} batch size {}".format(
         #           input.size(0), hidden_label, hx.size(0)))

        # TODO: Fix the first check

        if h1.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, h1.size(1), self.hidden_size))

        if h1.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, h2.size(1), self.hidden_size))


class MDRNNCell(MDRNNCellBase):
    r"""A (2D) Multi-Dimensional RNN cell with tanh or ReLU non-linearity.
      .. math::
          Normal 1D RNN:
          h' = \tanh(w_{ih} * x + b_{ih}  +  w_{hh} * h + b_{hh})
          2D RNN:
          h' = \tanh(w_{ih} * x + b_{ih}  +
                        w_{hh} * h + b_{hh}  +
                        w_{hh} * h + b_{hh}  +
                )

      If nonlinearity='relu', then ReLU is used in place of tanh.
      Args:
          input_size: The number of expected features in the input x
          hidden_size: The number of features in the hidden states h1 and h2
          bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
              Default: ``True``
          nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
      Inputs: input, hidden1, hidden2
          - **input** (batch, input_size): tensor containing input features
          - **hidden** (batch, hidden_size): tensor containing the initial hidden1
            state for each element in the batch.
          - **hidden2** (batch, hidden_size): tensor containing the initial hidden2
            state for each element in the batch.
      Outputs: h'
          - **h'** (batch, hidden_size): tensor containing the next hidden state
            for each element in the batch
      Attributes:
          weight_ih: the learnable input-hidden1 weights, of shape
              `(input_size x hidden_size)`
          weight_h1h: the learnable hidden1-hidden weights, of shape
              `(hidden_size x hidden_size)`
          weight_hh2: the learnable hidden2-hidden weights, of shape
              `(hidden_size x hidden_size)`
          bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
          bias_h1h: the learnable hidden1-hidden bias, of shape `(hidden_size)`
          bias_h2h: the learnable hidden2-hidden bias, of shape `(hidden_size)`
      Examples::
          >>> rnn = nn.RNNCell(10, 20)
          >>> input = Variable(torch.randn(6, 3, 10))
          >>> h1 = Variable(torch.randn(3, 20))
          >>> h2 = Variable(torch.randn(3, 20))
          >>> output = []
          >>> for i in range(6):
          ...     hx = rnn(input[i], h1, h2)
          ...     output.append(hx)
      """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(MDRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.ih = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h1h = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

        if not bias:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_h1h', None)
            self.register_parameter('bias_h2h', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h1, h2):
        self.check_forward_input(input)
        self.check_forward_hidden(input, h1, h2)
        if self.nonlinearity == "tanh":
            func = F.tanh
        elif self.nonlinearity == "relu":
            #func = self._backend.RNNReLUCell
            func = F.relu
        elif self.nonlinearity == "sigmoid":
            func = F.sigmoid
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        input_total = self.ih(input)
        h1h_total = self.h1h(h1)
        h2h_total = self.h1h(h2)
        #h_total = torch.add
        #total = torch.add(h_total, 1, input_total)
        total = input_total + h1h_total + h2h_total
        h_activation = func(total)
        #h_activation = torch.sigmoid(total)

        return h_activation

        #return func(
        #    input, hx,
        #    self.weight_ih, self.weight_hh,
        #    self.bias_ih, self.bias_hh,
        #)


def main():
    print("Testing the MultDimensionalRNN Cell... ")
    mdrnn = MDRNNCell(10, 5, nonlinearity="relu")
    input = Variable(torch.randn(6, 3, 10))

    # print("Input: " + str(input))

    h1 = Variable(torch.randn(3, 5))
    h2 = Variable(torch.randn(3, 5))
    output = []

    for i in range(6):
        print("iteration: " + str(i))
        h2 = mdrnn(input[i], h1, h2)
        print("h2: " + str(h2))
        output.append(h2)

    print(str(output))


if __name__ == "__main__":
    main()
