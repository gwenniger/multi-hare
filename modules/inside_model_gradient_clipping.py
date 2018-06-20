import torch


class InsideModelGradientClamping:
    #    CLAMPING_BOUND = 0.001
    #    CLAMPING_BOUND = 0.1
    CLAMPING_BOUND = 0.01
    # CLAMPING_BOUND = 1

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
    def clamp_grad_and_print(grad_input):
        # print("clamp_grad_and_pring - grad_input: " + str(grad_input))
        grad_output = grad_input.clamp(min=-InsideModelGradientClamping.CLAMPING_BOUND,
                                       max=InsideModelGradientClamping.CLAMPING_BOUND)
        # print("clamp_grad_and_pring - grad_output: " + str(grad_output))
        return grad_output

    # Note: register_gradient_clipping does have an effect. To see this effect though,
    # the value of CLAMPING_BOUND must be chosen rather small.
    # A good way to show the effect is to use the function "torch.nn.utils.clip_grad_norm_"
    # and print the total_norm it returns. It will be observed that when CLAMPING_BOUND
    # is made very small, and "register_gradient_clamping" is called for all tensor
    # variables in forward functions of the concerned modules, this has the effect of
    # making the total gradient norm very small
    @staticmethod
    def register_gradient_clamping(tensor: torch.Tensor):

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
            tensor.register_hook(lambda x: InsideModelGradientClamping.clamp_grad_and_print(x))

        else:
            raise RuntimeError("Error: register_gradient_clamping - not requiring gradient")

        return tensor
