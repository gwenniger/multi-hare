import torch
from torch.autograd import Variable
# See: https://stackoverflow.com/questions/24197970/pycharm-import-external-library
import warpctc_pytorch
import torch.optim as optim
from util.tensor_utils import TensorUtils

# This test class aims to clarify how to use the waprctc_pytorch interface
# by different examples.
# These are some important
# lessons learned from how to call
# ctc_loss(probs, labels, probs_sizes, label_sizes).
# These tests are mostly ports of the examples from
# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
#
# but also contain some variants that clarify some subtleties.
#
# Some lessons learned are:
#
# 1. probs must indeed be a tensor of dimension  seqLength x batchSize x alphabet_size
#    For example:
#           blank alphabet_1 alphabet_2 alphabet_3 alphabet_4
#    probs: tensor([[[  0.,   0.,   0.,   0.,   0.],   # example 1 seq element 1
#          [  1.,   2.,   3.,   4.,   5.]],            # example 2 seq element 1
#
#         [[  0.,   0.,   0.,   0.,   0.],             # example 1 seq element 2
#          [  6.,   7.,   8.,   9.,  10.]],            # example 2 seq element 2
#
#         [[  0.,   0.,   0.,   0.,   0.],             # example 1 seq element 3
#          [ 11.,  12.,  13.,  14.,  15.]],            # example 2 seq element 3
#
#         [[  0.,   0.,   0.,   0.,   0.],             # example 1 seq element 4
#          [  0.,   0.,   0.,   0.,   0.]]])           # example 2 seq element 4
#  probs.size(): torch.Size([4, 2, 5])
# 2. labels contains all the labels concatenated in a single 1-dimensional Integer
#    tensor. This tensor does not contain blank (=0) labels.
#    See: https://github.com/SeanNaren/warp-ctc/issues/29
# 3. label_sizes specifies the number of elements of each of the label sequences.
#    This is necessary exactly since labels contains all label sequences, for
#    each of the batch items/examples,  concatenated together. So to know how to split
#    up the concatenated sequence into the sequences that belong to each of the examples
#    in the batch, the elements of label_sizes are used.
# 4. probs_sizes is necessary, since not all the examples in the probs tensor may be
#    of the same real size. If one of the examples is shorter, it is added to the
#    batch, by adding additional rows int the sequence (first) dimension as zeros
#    Note that it is even allowed to add arbitrary many zero rows to each of the
#    examples, and because of probs_sizes these will not change the result. However,
#    there is no reason to do so. Note that it is important that the padding is added
#    at the bottom, not at the top. One of the tests below checks that this is how
#    it should be done, and that padding on the wrong side indeed causes the results
#    to chance (since that causes the padding to be interpreted as input, and the
#    real input as padding).
#
#


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


#### These examples mainly show that weird inputs do not necessarily cause the
# program to crash in an interpretable way, but instead yield results that may
# be hard to interpret for someone who does not exactly now how to use the
# warpctc_pytorch interface.
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


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
# https://github.com/baidu-research/warp-ctc
# Seperately the batch items work ok, but when joined together the scores become different
def test_ctc_loss_probabilities_match_labels_third_baidu_example():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    # https://stackoverflow.com/questions/48915810/pytorch-contiguous
    probs = torch.FloatTensor([
                        [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5],  [-5, -4, -3, -2, -1]],
                        [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10], [-10, -9, -8, -7, -6]],
                        [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15],  [-15, -14, -13, -12, -11]]
                        ])# .contiguous() # contiguous is just for performance, does not change results

    # probs = torch.FloatTensor([
    #     [[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6], [-15, -14, -13, -12, -11]]
    # ]). \
    #    transpose(0, 1).contiguous()

    print("test_ctc_loss_probabilities_match_labels_third_baidu_example - probs: " + str(probs))

    print("test_ctc_loss_probabilities_match_labels_third_baidu_example - probs.size(): " + str(probs.size()))

    # labels = Variable(torch.IntTensor([ [1, 0], [3, 3], [2, 3]]))
    # See: https://github.com/SeanNaren/warp-ctc/issues/29
    # All label sequences are concatenated, without blanks/padding,
    # and label sizes lists the sizes without padding
    labels = Variable(torch.IntTensor([1, 3, 3, 2, 3]))
    # labels = Variable(torch.IntTensor([2, 3]))
    #labels = Variable(torch.IntTensor([3, 3]))
    # Labels sizes should be equal to number of labels
    label_sizes = Variable(torch.IntTensor([1, 2, 2]))
    #label_sizes = Variable(torch.IntTensor([2]))
    # This one must be equal to the number of probabilities to avoid a crash
    probs_sizes = Variable(torch.IntTensor([1, 3, 3]))
    # probs_sizes = Variable(torch.IntTensor([3]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    # cost: tensor([ 7.3557]) as in the Baidu tutorial, second example
    print("cost: " + str(cost))
    expected_cost_tensor = torch.FloatTensor([13.904030799865723])
    print("zeros_tensor: " + str(expected_cost_tensor))
    if not TensorUtils.tensors_are_equal(expected_cost_tensor, cost):
        raise RuntimeError("Error: cost expected to be " + str(expected_cost_tensor) +
                           "but was:" + str((float(cost))))
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))

    print(">>> Success: test_ctc_loss_probabilities_match_labels_third_baidu_example")


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
# https://github.com/baidu-research/warp-ctc
# Seperately the batch items work ok, but when joined together the scores become different
def test_ctc_loss_probabilities_match_labels_third_baidu_example_variant():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    probs = torch.FloatTensor([
                        [[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [-5, -4, -3, -2, -1]],
                        [[6, 7, 8, 9, 10], [0, 0, 0, 0, 0], [-10, -9, -8, -7, -6]],
                        [[11, 12, 13, 14, 15], [0, 0, 0, 0, 0], [-15, -14, -13, -12, -11]]
                        ]).contiguous()

    # probs = torch.FloatTensor([
    #     [[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6], [-15, -14, -13, -12, -11]]
    # ]). \
    #    transpose(0, 1).contiguous()

    print("probs.size(): " + str(probs.size()))

    # labels = Variable(torch.IntTensor([ [1, 0], [3, 3], [2, 3]]))
    # See: https://github.com/SeanNaren/warp-ctc/issues/29
    # All label sequences are concatenated, without blanks/padding,
    # and label sizes lists the sizes without padding
    labels = Variable(torch.IntTensor([3, 3, 1, 2, 3]))
    # labels = Variable(torch.IntTensor([2, 3]))
    #labels = Variable(torch.IntTensor([3, 3]))
    # Labels sizes should be equal to number of labels
    label_sizes = Variable(torch.IntTensor([2, 1,  2]))
    #label_sizes = Variable(torch.IntTensor([2]))
    # This one must be equal to the number of probabilities to avoid a crash
    probs_sizes = Variable(torch.IntTensor([3, 1, 3]))
    # probs_sizes = Variable(torch.IntTensor([3]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    # cost: tensor([ 7.3557]) as in the Baidu tutorial, second example
    print("cost: " + str(cost))
    expected_cost_tensor = torch.FloatTensor([13.904030799865723])
    print("zeros_tensor: " + str(expected_cost_tensor))
    if not TensorUtils.tensors_are_equal(expected_cost_tensor, cost):
        raise RuntimeError("Error: cost expected to be " + str(expected_cost_tensor) +
                           "but was:" + str((float(cost))))
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))

    print(">>> Success: test_ctc_loss_probabilities_match_labels_third_baidu_example_variant")


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
# https://github.com/baidu-research/warp-ctc
# Seperately the batch items work ok, but when joined together the scores become different
def test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    # https://stackoverflow.com/questions/48915810/pytorch-contiguous
    probs = torch.FloatTensor([
                        [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
                        [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10]],
                        [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15]]
                        ]) # .contiguous() # contiguous is just for performance, does not change results

    print("probs.size(): " + str(probs.size()))

    # labels = Variable(torch.IntTensor([ [1, 0], [3, 3], [2, 3]]))
    # See: https://github.com/SeanNaren/warp-ctc/issues/29
    # IMPORTANT !!!: All label sequences are concatenated, without blanks/padding,
    # and label sizes lists the sizes without padding
    labels = Variable(torch.IntTensor([1, 3, 3]))
    # Labels sizes should be equal to number of labels. Because labels are
    # concatenated, the label sizes essentially instructs where the sequence
    # boundaries are!
    label_sizes = Variable(torch.IntTensor([1, 2]))
    # Prob_sizes instructs on the number of real probabilities, distinguishing
    # real probabilities from padding
    # Padding should presumably
    # (looking at https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md)
    # be at the bottom, but this should be checked
    probs_sizes = Variable(torch.IntTensor([1, 3]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    # cost: tensor([ 7.3557]) as in the Baidu tutorial, second example
    print("cost: " + str(cost))
    expected_cost_tensor = torch.FloatTensor([8.965181350708008])
    print("zeros_tensor: " + str(expected_cost_tensor))
    if not TensorUtils.tensors_are_equal(expected_cost_tensor, cost):
        raise RuntimeError("Error: cost expected to be " + str(expected_cost_tensor) +
                           "but was:" + str((float(cost))))
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))

    print(">>> Success: test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two")


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
# https://github.com/baidu-research/warp-ctc
# Seperately the batch items work ok, but when joined together the scores become different
def test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two_extra_padding():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    # https://stackoverflow.com/questions/48915810/pytorch-contiguous
    probs = torch.FloatTensor([
                        [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
                        [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10]],
                        [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15]],
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],   # Extra padding is added at the bottom
                        ]) # .contiguous() # contiguous is just for performance, does not change results

    print("probs.size(): " + str(probs.size()))

    # labels = Variable(torch.IntTensor([ [1, 0], [3, 3], [2, 3]]))
    # See: https://github.com/SeanNaren/warp-ctc/issues/29
    # IMPORTANT !!!: All label sequences are concatenated, without blanks/padding,
    # and label sizes lists the sizes without padding
    labels = Variable(torch.IntTensor([1, 3, 3]))
    # Labels sizes should be equal to number of labels. Because labels are
    # concatenated, the label sizes essentially instructs where the sequence
    # boundaries are!
    label_sizes = Variable(torch.IntTensor([1, 2]))
    # Prob_sizes instructs on the number of real probabilities, distinguishing
    # real probabilities from padding
    # Padding should presumably
    # (looking at https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md)
    # be at the bottom, but this should be checked
    probs_sizes = Variable(torch.IntTensor([1, 3]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    # cost: tensor([ 7.3557]) as in the Baidu tutorial, second example
    print("cost: " + str(cost))
    # Since padding has been added on the right side (bottom)
    # the results are not expected to change from the example without padding
    expected_cost_tensor = torch.FloatTensor([8.965181350708008])
    print("zeros_tensor: " + str(expected_cost_tensor))
    if not TensorUtils.tensors_are_equal(expected_cost_tensor, cost):
        raise RuntimeError("Error: cost expected to be " + str(expected_cost_tensor) +
                           "but was:" + str((float(cost))))
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))

    print(">>> Success: test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two_extra_padding")


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
# https://github.com/baidu-research/warp-ctc
# Seperately the batch items work ok, but when joined together the scores become different
def test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two_extra_padding_wrong_side():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")
    # https://stackoverflow.com/questions/48915810/pytorch-contiguous
    probs = torch.FloatTensor([
                        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],  # Extra padding is added at the top, which is wrong
                        [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
                        [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10]],
                        [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15]],
                        ]) # .contiguous() # contiguous is just for performance, does not change results

    print("probs.size(): " + str(probs.size()))

    # labels = Variable(torch.IntTensor([ [1, 0], [3, 3], [2, 3]]))
    # See: https://github.com/SeanNaren/warp-ctc/issues/29
    # IMPORTANT !!!: All label sequences are concatenated, without blanks/padding,
    # and label sizes lists the sizes without padding
    labels = Variable(torch.IntTensor([1, 3, 3]))
    # Labels sizes should be equal to number of labels. Because labels are
    # concatenated, the label sizes essentially instructs where the sequence
    # boundaries are!
    label_sizes = Variable(torch.IntTensor([1, 2]))
    # Prob_sizes instructs on the number of real probabilities, distinguishing
    # real probabilities from padding
    # Padding should presumably
    # (looking at https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md)
    # be at the bottom, but this should be checked
    probs_sizes = Variable(torch.IntTensor([1, 3]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    # cost: tensor([ 7.3557]) as in the Baidu tutorial, second example
    print("cost: " + str(cost))
    # Since padding has been added to the wrong side (top instead of bottom)
    # the results are now expected to change
    no_longer_expected_cost_tensor = torch.FloatTensor([8.965181350708008])
    print("zeros_tensor: " + str(no_longer_expected_cost_tensor))
    if TensorUtils.tensors_are_equal(no_longer_expected_cost_tensor, cost):
        raise RuntimeError("Error: cost expected to be not equal to " + str(no_longer_expected_cost_tensor) +
                           "but was:" + str((float(cost))))
    cost.backward()
    print("cost: " + str(cost))
    print("update probabilities...")
    optimizer.step()
    print("probs: " + str(probs))

    print(">>> Success: test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two_extra_padding_wrong_side")


# https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
def test_ctc_loss_probabilities_match_labels_three():

    ctc_loss = warpctc_pytorch.CTCLoss()
    print("expected shape of seqLength x batchSize x alphabet_size")

    # Gives no loss
    probs = torch.FloatTensor([[[0, 100, 0, 0, 83],
                                [0, 0, 100, 0, 0],
                                [0, 0, 0, 100, 0]]]).\
        transpose(0, 1).contiguous()

    # # Gives small loss
    # probs = torch.FloatTensor([[[0, 100, 0, 0, 84],
    #                             [0, 0, 100, 0, 0],
    #                             [0, 0, 0, 100, 0]]]). \
    #     transpose(0, 1).contiguous()

    print("probs.size(): " + str(probs.size()))

    # No loss
    # labels = Variable(torch.IntTensor([1, 2, 3]))
    # Also no loss (possibly because not possible!)
    # becomes effectively 2-2-2-2 which is length 6!
    # labels = Variable(torch.IntTensor([2, 2, 2, 2]))
    # labels (becomes 2-2) (Why is loss also zero?)
    labels = Variable(torch.IntTensor([1, 1, 1]))
    # Labels sizes should be equal to the number of labels in the example
    label_sizes = Variable(torch.IntTensor([3]))
    # This one must be equal to the number of probabilities to avoid a crash
    probs_sizes = Variable(torch.IntTensor([3]))
    probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
    optimizer = optim.SGD(list([probs]), lr=0.001)
    print("probs: " + str(probs))
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    # cost: tensor([ 7.3557]) as in the Baidu tutorial, second example
    print("cost: " + str(cost))
    expected_cost_tensor = torch.FloatTensor([0])
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
    # CTC Beam decoder: https://github.com/joshemorris/pytorch-ctc

    # test_ctc_loss()
    # test_ctc_loss_two()
    # test_ctc_loss_probabilities_match_labels()
    test_ctc_loss_probabilities_match_labels_second_baidu_example()
    test_ctc_loss_probabilities_match_labels_third_baidu_example()
    test_ctc_loss_probabilities_match_labels_third_baidu_example_variant()
    test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two()
    test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two_extra_padding()
    test_ctc_loss_probabilities_match_labels_third_baidu_example_variant_two_extra_padding_wrong_side()
    test_ctc_loss_probabilities_match_labels_three()


if __name__ == "__main__":
    main()
