"""
This class implements a mock/fake CTC loss function, that can be used for debugging purposes.
Specifically it can potentially be used to pinpoint the "real" CTC loss as a cause for certain problems
such as leaking file descriptors in DataLoader, see:
https://github.com/pytorch/pytorch/issues/65198
https://github.com/pytorch/pytorch/issues/11201
https://github.com/pytorch/pytorch/issues/973

"""
import torch

# class MockCTCLoss:
#     def __str__(self):
#         pass
#
#     def forward(self):
#         pass

class MockWarpCTCLossInterface:

    def __init__(self):
        pass

    def compute_ctc_loss(self, probabilities, labels_row_tensor, batch_size: int,
                         width_reduction_factor: int):
        #return MockCTCLoss()
        # Use a simple zeros tensor with just one elelment. All the ope
        return torch.zeros(1, requires_grad=True)
