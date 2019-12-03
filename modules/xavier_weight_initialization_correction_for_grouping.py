import torch
import math

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class XavierWeightInitializationCorrectionForGrouping:

    @staticmethod
    def correct_xavier_uniform_initialized_weights_for_grouping(layer_weights: torch.Tensor, groups: int):
        """
        This method corrects the weight of a layer weight-matrix, where the layer computes the output
        for multiple groups, such that the weights for each group should have been initialized as if
        they were separate layers. Looking at the formula for Xavier weight initialization, it turns out
        that by re-multiplying the Xavier-initialized weights with a factor sqrt(groups), we get the
        correctly initialized weights.

        :param layer_weights: The layer weight matrix
        :param groups: The number of independent groups, which can be considered as number of virtual layers
        :return: Nothing, the weight correction is done in-place on the input
        """

        correction_factor = math.sqrt(groups)
        print(">>>Compensating Xavier weight initialization "
              "for a layer that parallelly computes the result for " + str(groups) + " groups"
              ", by re-multiplying weight by factor sqrt(" + str(groups) + " )= " + str(correction_factor))
        # We multiply the weights again with 2 = sqrt(4) after the initial
        # Xavier uniform initialization. The motivation is that Xavier uniform
        # initialization initializes weights to :
        # Wi,j = U(-sqrt(6/(m+n), sqrt(6/(m+n)) with m the fan-in and n the fan-out.
        # However, in this case we want to normalize as if we were really having four
        # separate feed-forward layers that where al separately initialized, then summed.
        # This means we have to compensate for the fact that m is four times too big
        # during the Xavier initialization. This is done by multiplying the weights
        # again with sqrt(4) = 2, since:
        # sqrt(6/((m/4)+n) = sqrt(6*4/(m+n) = sqrt(4) * sqrt(6/(m+n) =
        # 2 * sqrt(6/(m+n)
        # This should facilitate better learning, since we empirically observe that
        # without this compensation the initial learning is very slow.
        # See also: https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        print("Before compensation: " + str(layer_weights.norm()))
        with torch.no_grad():
            layer_weights.mul_(correction_factor)
        print("After compensation: " + str(layer_weights.norm()))