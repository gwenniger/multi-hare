import torch
from torch.autograd import Variable
# See: https://stackoverflow.com/questions/24197970/pycharm-import-external-library
import warpctc_pytorch
import torch.optim as optim
from util.tensor_utils import TensorUtils


def test_ctc_loss():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
    print("probs.size(): " + str(probs.size()))
    labels = Variable(torch.IntTensor([1, 2]))
    label_sizes = Variable(torch.IntTensor([2]))
    probs_sizes = Variable(torch.IntTensor([2]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001, momentum=0.9, weight_decay=1e-5)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))

# Baidu simple example
# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
# Seems to make more sense
def test_ctc_loss_two():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")

    # gives cost inf
    #probs = torch.FloatTensor([[[0, 0, 0, 0, 1000]]]).transpose(0, 1).contiguous()

    # gives cost 0
    # probs = torch.FloatTensor([[[0, 100, 0, 0, 0]]]).transpose(0, 1).contiguous()

    ## Everything equally likely: gives cost 1.6094
    #probs = torch.FloatTensor([[[0, 0, 0, 0, 0]]]).transpose(0, 1).contiguous()

    # Everything equally likely: gives cost 1.6094
    probs = torch.FloatTensor([[[1, 1, 1, 1, 1]]]).transpose(0, 1).contiguous()

    print("probs.size(): " + str(probs.size()))
    labels = Variable(torch.IntTensor([1]))
    label_sizes = Variable(torch.IntTensor([1]))
    probs_sizes = Variable(torch.IntTensor([1]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001, momentum=0.9, weight_decay=1e-5)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
def test_ctc_loss_probabilities_match_labels():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    probs = torch.FloatTensor([[[0.9, 1.0, 0.0, 0.0],
                                [0.1, 0.0, 1.0, 1.0]]]).\
        transpose(0, 1).contiguous()

    print("probs.size(): " + str(probs.size()))
    # No cost
    labels = Variable(torch.IntTensor([1, 1, 2, 1]))
    # No cost
    labels = Variable(torch.IntTensor([1, 1, 1, 1]))
    # Cost
    labels = Variable(torch.IntTensor([1, 2, 2, 1]))
    # No cost
    labels = Variable(torch.IntTensor([1, 1]))
    # No cost
    labels = Variable(torch.IntTensor([2, 2]))
    # Crash (Apparently must be minimally 2 elements)
    labels = Variable(torch.IntTensor([2]))
    # No cost
    labels = Variable(torch.IntTensor([3, 3]))
    # FIXME What is going on here?!?

    label_sizes = Variable(torch.IntTensor([2]))
    # This one must be equal to the number of probabilities to avoid a crash
    probs_sizes = Variable(torch.IntTensor([2]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    print("cost: " + str(cost))
    zero_tensor = torch.zeros(1)
    print("zeros_tensor: " + str(zero_tensor))
    if not TensorUtils.tensors_are_equal(zero_tensor, cost):
        raise RuntimeError("Error: loss expected to be zero, since probabilities " +
                           "are maximum for the right labels, but not the case")
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
def test_ctc_loss_probabilities_match_labels_second_baidu_example():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    probs = torch.FloatTensor([[[1, 2, 3, 4, 5],
                                [6, 7, 8, 9, 10],
                                [11, 12, 13, 14, 15]]]).\
        transpose(0, 1).contiguous()

    print("probs.size(): " + str(probs.size()))

    labels = Variable(torch.IntTensor([3, 3]))
    # Labels sizes should be equal to number of labels
    label_sizes = Variable(torch.IntTensor([2]))
    # This one must be equal to the number of probabilities to avoid a crash
    probs_sizes = Variable(torch.IntTensor([3]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    # cost: tensor([ 7.3557]) as in the Baidu tutorial, second example
    print("cost: " + str(cost))
    expected_cost_tensor = torch.FloatTensor([7.355742931365967])
    print("zeros_tensor: " + str(expected_cost_tensor))
    if not TensorUtils.tensors_are_equal(expected_cost_tensor, cost):
        raise RuntimeError("Error: cost expected to be " + str(expected_cost_tensor) +
                           "but was:" + str((float(cost))))
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))


def main():
    # test_ctc_loss()
    # test_ctc_loss_two()
    # test_ctc_loss_probabilities_match_labels()
    test_ctc_loss_probabilities_match_labels_second_baidu_example()

if __name__ == "__main__":
    main()
