import torch
from torch.nn.modules.module import Module
from torch.autograd.function import Function

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


# https://discuss.pytorch.org/t/defining-backward-function-in-nn-module/5047
class GradientClampingFunction(Function):
    def forward(self, input, parameters):
        self.saved_for_backward = [input, parameters]
        # output = [do something with input and parameters]
        output = input
        print("GradientClampingFunction.forward - output: " + str(output))
        return output

    def backward(self, grad_output):
        input, parameters = self.saved_for_backward
        # grad_input = [derivate forward(input) wrt parameters] * grad_output
        grad_input = grad_output.clamp(min=-10,
                                       max=10)
        return grad_input


class GradientClampedModule(Module):

    def __init__(self, module):
        super(GradientClampedModule, self).__init__()
        self.module = module

        # https://github.com/pytorch/pytorch/issues/7040
        # https://github.com/pytorch/pytorch/issues/598
        # self.module.register_backward_hook(lambda module,
        #                                           grad_i: GradientClampedModule.clamp_gradient_tuples(grad_i))
        self.weight = module.weight
        self.bias = module.bias
        self.gradient_clamping_function = GradientClampingFunction()
        self.parameters = module.parameters

    # See: https://discuss.pytorch.org/t/register-backward-hook-on-nn-sequential/472/5
    @staticmethod
    def clamp_gradient_tuples(grad_i):
            return tuple([grad_i[index].clamp(min=-10, max=10) for index in range(0, len(grad_i))])

    def forward(self, input):
        # print("In GradientClampedModule forward...")
        module_result = self.module(input)
        # return self.gradient_clamping_function(module_result, self.weight)
        return module_result

    def backward(self, input):
        module_gradient = self.module.backward(input)
        #output = self.gradient_clamping_function(module_gradient)
        torch.nn.utils.clip_grad_value_(self.module.weight)
        torch.nn.utils.clip_grad_value_(self.module.bias)
        output = module_gradient
        return output
