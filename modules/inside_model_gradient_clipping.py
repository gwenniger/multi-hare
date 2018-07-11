import torch
import util.tensor_utils
import inspect

class InsideModelGradientClamping:
    # CLAMPING_BOUND = 0.05
    # CLAMPING_BOUND = 0.1
    # CLAMPING_BOUND = 0.000001
    # CLAMPING_BOUND = 0.01
    # CLAMPING_BOUND = 10
    # https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
    # See: https://keras.io/optimizers/

    # See: https://github.com/t-vi/pytorch-tvmisc/blob/master/misc/graves_handwriting_generation.ipynb
    # A default clamping bound of 10 seems to work well for (MD)LSTM internal states
    # a higher bound of e.g. 100 can be used for linear layers etc
    # Choosing the clamping bounds too low seems to potentially slow down learning.
    CLAMPING_BOUND = 10

    # This method registers a gradient clamping hook for the gradient of the
    # weight tensor, which will clamp/clip the gradient to the clamping range.
    # This is somewhat similar to gradient clipping on the loss function,
    # but doing the clamping during the back propagation for each weight
    # tensor separately has important advantage over clipping the loss function
    # using the "torch.nn.utils.clip_grad_norm_" method.
    # Why?
    #    Clipping the gradient norm with the "torch.nn.utils.clip_grad_norm_"
    #    method is done after the entire back_propagation has been completed.
    #    Furthermore, everything is then scaled by same rescaling factor based
    #    on the gradient norm and the maximum permitted norm. But this means
    #    that perfectly fine gradient components will become very small if there
    #    is one component in the gradient that makes the gradient norm very big.
    #    This can be expected to be counter productive to effective learning.
    #
    # Would "torch.nn.clip_grad_value_(parameters, clip_value)" not have the
    # same effect?
    #
    #   Clipping with "clip_grad_value_(parameters, clip_value)" still clips only
    #   after the full gradient has been computed using the "loss.backward" method.
    #   As such it may or may not give similar results, this is an empirical matter.
    #
    # From https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/15:
    #
    #
    #
    #

    @staticmethod
    def is_bad_grad(grad_output):
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    @staticmethod
    def clamp_grad(grad_input, clamping_bound, variable_name: str):

        grad_output = grad_input.clamp(min=-clamping_bound,
                                       max=clamping_bound)

        # if variable_name == "mdlstm - activation_column":
        #     print("clamping gradient - " + variable_name)
        #     print("clamp_grad_and_print - grad_input: " + str(grad_input))
        #     print("clamp_grad_and_print - grad_output: " + str(grad_output))


        if InsideModelGradientClamping.is_bad_grad(grad_input) :
            print("is_bad_grad - grad_input: " + str(grad_input))
            ##not util.tensor_utils.TensorUtils.tensors_are_equal(grad_input, grad_output):
            # https://stackoverflow.com/questions/900392/getting-the-caller-function-name-inside-another-function-in-python
            print("clamping gradient - " + variable_name)
            print("clamp_grad_and_print - grad_input: " + str(grad_input))
            print("clamp_grad_and_print - grad_output: " + str(grad_output))

        if InsideModelGradientClamping.is_bad_grad(grad_output):
            print("is_bad_grad - grad_output: " + str(grad_output))
            ##not util.tensor_utils.TensorUtils.tensors_are_equal(grad_input, grad_output):
            # https://stackoverflow.com/questions/900392/getting-the-caller-function-name-inside-another-function-in-python
            print("clamping gradient - " + variable_name)
            print("clamp_grad_and_print - grad_input: " + str(grad_input))
            print("clamp_grad_and_print - grad_output: " + str(grad_output))
        return grad_output

    @staticmethod
    def clamp_grad_and_print(grad_input, clamping_bound):
        print("clamp_grad_and_print - grad_input: " + str(grad_input))
        grad_output = grad_input.clamp(min=-clamping_bound,
                                       max=clamping_bound)
        print("clamp_grad_and_print - grad_output: " + str(grad_output))
        return grad_output

    # Note: register_gradient_clipping does have an effect. To see this effect though,
    # the value of CLAMPING_BOUND must be chosen rather small.
    # A good way to show the effect is to use the function "torch.nn.utils.clip_grad_norm_"
    # and print the total_norm it returns. It will be observed that when CLAMPING_BOUND
    # is made very small, and "register_gradient_clamping" is called for all tensor
    # variables in forward functions of the concerned modules, this has the effect of
    # making the total gradient norm very small
    @staticmethod
    def register_gradient_clamping(tensor: torch.Tensor, clamping_bound, print_gradient: bool, variable_name: str):

        # See: https://discuss.pytorch.org/t/gradient-clipping/2836/9
        # See: https://github.com/DingKe/pytorch_workplace/blob/master/rnn/modules.py#L122
        # Not sure why this is needed (does the hook not exist outside the function scope
        # if it is added directly to the function argument?)
        #tensor_temp = tensor.expand_as(tensor)

        if tensor.requires_grad:
            # Register a hook that will take care of clipping/clamping the gradient of the
            # weights to an appropriate weight, to avoid numerical problems
            #tensor.register_hook(lambda x:
            #                            x.clamp(min=-InsideModelGradientClamping.CLAMPING_BOUND,
            #                            max=InsideModelGradientClamping.CLAMPING_BOUND)
            #                     )
            if print_gradient:
                tensor.register_hook(lambda x: InsideModelGradientClamping.
                                     clamp_grad_and_print(x, clamping_bound))
            else:
                tensor.register_hook(lambda x: InsideModelGradientClamping.
                                     clamp_grad(x, clamping_bound, variable_name))

        # In evaluation mode no gradient will be required
        # else:
        #     raise RuntimeError("Error: register_gradient_clamping - not requiring gradient")

        return tensor

    @staticmethod
    def register_gradient_clamping_default_clamping_bound(tensor: torch.Tensor, variable_name: str):
        return InsideModelGradientClamping.register_gradient_clamping(tensor,
                                                                      InsideModelGradientClamping.CLAMPING_BOUND,
                                                                      False, variable_name)
