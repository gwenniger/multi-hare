import torch
from torch.autograd import Variable
# See: https://stackoverflow.com/questions/24197970/pycharm-import-external-library
import warpctc_pytorch
import torch.optim as optim


def test_ctc_loss():

    ctc_loss = warpctc_pytorch.CTCLoss()
    # expected shape of seqLength x batchSize x alphabet_size
    probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
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



def main():
    test_ctc_loss()


if __name__ == "__main__":
    main()
