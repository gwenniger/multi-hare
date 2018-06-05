import warpctc_pytorch
import torch.tensor
from util.utils import Utils
from torch.autograd import Variable


class WarpCTCLossInterface:

    def __init__(self):
        self.ctc_loss = warpctc_pytorch.CTCLoss()


    @staticmethod
    def create_warp_ctc_loss_interface():
        return WarpCTCLossInterface()

    # This method takes a tensor of size batch_size * sequence_length
    # that is, every row is a sequence of labels for an example
    # It then returns a one-dimensional label tensor formed by
    # concatenating all the row tensors
    @staticmethod
    def create_one_dimensional_labels_tensor(labels_row_tensor):
        labels_one_dimensional = labels_row_tensor.view(-1)
        # print("labels_one_dimensional: " + str(labels_one_dimensional))
        return labels_one_dimensional

    @staticmethod
    def create_sequence_lengths_specification_tensor_all_same_length(labels_row_tensor):
        number_of_examples = labels_row_tensor.size(0)
        sequence_length = labels_row_tensor.size(1)
        result = torch.IntTensor([sequence_length])
        for i in range(1, number_of_examples):
            result = torch.cat((result, torch.IntTensor([sequence_length])), 0)

        return result

    @staticmethod
    def create_probabilities_lengths_specification_tensor_all_same_length(probabilities):
        number_of_examples = probabilities.size(0)
        sequence_length = probabilities.size(1)
        result = torch.IntTensor([sequence_length])
        for i in range(1, number_of_examples):
            result = torch.cat((result, torch.IntTensor([sequence_length])), 0)

        return result

    def compute_ctc_loss_version_two(self, probabilities, labels_row_tensor):
        ctc_loss = warpctc_pytorch.CTCLoss()

        #probs = torch.FloatTensor([
        #    [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [-5, -4, -3, -2, -1]],
        #    [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10], [-10, -9, -8, -7, -6]],
        #    [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15], [-15, -14, -13, -12, -11]]
        #])

        probs = probabilities

        print("test_ctc_loss_probabilities_match_labels_third_baidu_example - probs: " + str(probs))

        print("test_ctc_loss_probabilities_match_labels_third_baidu_example - probs.size(): " + str(probs.size()))

        # labels = Variable(torch.IntTensor([ [1, 0], [3, 3], [2, 3]]))
        # See: https://github.com/SeanNaren/warp-ctc/issues/29
        # All label sequences are concatenated, without blanks/padding,
        # and label sizes lists the sizes without padding
        labels = Variable(torch.IntTensor([1, 3, 3, 2, 3]))
        # labels = Variable(torch.IntTensor([2, 3]))
        # labels = Variable(torch.IntTensor([3, 3]))
        # Labels sizes should be equal to number of labels
        label_sizes = Variable(torch.IntTensor([1, 2, 2]))
        # label_sizes = Variable(torch.IntTensor([2]))
        # This one must be equal to the number of probabilities to avoid a crash
        probs_sizes = Variable(torch.IntTensor([1, 3, 3]))
        # probs_sizes = Variable(torch.IntTensor([3]))
        probs = Variable(probs, requires_grad=True)  # tells autograd to compute gradients for probs
        print("probs: " + str(probs))

        if Utils.use_cuda():
            probs = probs.cuda()
            device = probs.get_device()
            ctc_loss = ctc_loss.cuda()
            # labels = labels.cuda()
            # label_sizes = label_sizes.cuda()
            # probs_sizes = probs_sizes.cuda()


        loss = ctc_loss(probs, labels, probs_sizes, label_sizes)
        print("loss: " + str(loss))

        return loss

    # Computes the ctc_loss for a probabilities tensor of dimensions:
    # 0: batch size, 1: sequence length, 2: number of symbol types + 1 (for blank)

    def compute_ctc_loss(self, probabilities, labels_row_tensor, batch_size):

        number_of_labels = labels_row_tensor.view(-1).size(0)
        non_zeros = torch.nonzero(labels_row_tensor.view(-1).data).squeeze()
        number_of_zero_labels = number_of_labels - non_zeros.size(0)

        # A sanity check to make sure the labels_row_tensor does not contain zeros,
        # which was an error in past usage
        if number_of_zero_labels != 0:
            raise RuntimeError("Error: label_row_tensor contains zero labels" +
                               " only non-zero labels are allowed, since the 0 " +
                               "label is reserved for blanks")

        labels = Variable(WarpCTCLossInterface.
                          create_one_dimensional_labels_tensor(labels_row_tensor))
        label_sizes = Variable(WarpCTCLossInterface.\
            create_sequence_lengths_specification_tensor_all_same_length(labels_row_tensor))
        probabilities_sizes = Variable(WarpCTCLossInterface.\
            create_probabilities_lengths_specification_tensor_all_same_length(probabilities))

        # The ctc_loss interface expects the second dimension to be the batch size,
        # so the first and second dimension must be swapped
        probabilities_batch_second_dimension = probabilities.transpose(0, 1).contiguous()

        if Utils.use_cuda():
            device = probabilities.get_device()
            self.ctc_loss = self.ctc_loss.to(device)
            # self.ctc_loss = self.ctc_loss.cuda()
            # https://discuss.pytorch.org/t/which-device-is-model-tensor-stored-on/4908/7
            # device = probabilities.get_device()
            # Causes "Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)"
            # labels = labels.cuda()

            #probabilities_batch_second_dimension = torch.zeros(probabilities_batch_second_dimension.size(0),
            #                                                   probabilities_batch_second_dimension.size(1),
            #                                                   probabilities_batch_second_dimension.size(2),
            #                                                   requires_grad=True
            #                                                )

            probabilities_batch_second_dimension = probabilities_batch_second_dimension.cuda()

            # probabilities_batch_second_dimension = probabilities_batch_second_dimension.to(device)

            # print("probabilities_batch_second_dimension.requires_grad:" +
            #      str(probabilities_batch_second_dimension.requires_grad))

            #probabilities_sizes = Variable(torch.IntTensor([1, 3, 3]))
            #probabilities_sizes = probabilities_sizes.to(device)

        # print("probabilities_batch_second_dimension: " + str(probabilities_batch_second_dimension))

        # print("probabilities_sizes: " + str(probabilities_sizes))

        # print(">>> compute_ctc_loss - probabilities_batch_second_dimension.size(): "
        #      + str(probabilities_batch_second_dimension.size()))
        # print(">>> compute_ctc_loss - labels.size(): " + str(labels.size()))
        # print(">>> compute_ctc_loss - label_sizes.size(): " + str(label_sizes.size()))
        # print(">>> compute_ctc_loss - probabilities_sizes.size(): " + str(probabilities_sizes.size()))
        # print("label_sizes: " + str(label_sizes))
        # print("labels: " + str(labels))
        # print("probabilities_sizes: " + str(probabilities_sizes))

        # Sanity check: the batch size must be the right dimension of the probabilities
        # tensor, otherwise the ctc_loss function will give wrong results and or
        # crash.
        if probabilities_batch_second_dimension.size(1) != batch_size:
            raise RuntimeError("Error: the second dimension of probabilities_batch_second_dimension" +
                               "should equal batch_size " + str(batch_size) + " but is " +
                               str(probabilities_batch_second_dimension.size(1))
                               )

        loss = self.ctc_loss(probabilities_batch_second_dimension, labels, probabilities_sizes, label_sizes)

        # print(">>> compute_ctc_loss - loss: " + str(loss))

        return loss


def test_warp_ctc_loss_interface():
    probs = torch.FloatTensor([
        [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [-5, -4, -3, -2, -1]],
        [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10], [-10, -9, -8, -7, -6]],
        [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15], [-15, -14, -13, -12, -11]]
    ])

    probs = probs.transpose(0, 1)
    probs = probs.cuda()

    probs = Variable(probs, requires_grad=True)

    # Weirdly enough it is OK if the probabilities are on the GPU, but if the labels are on the GPU
    # things crash with:
    # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)

    # It does not work with labels using cuda
    # labels_row_tensor = Variable(torch.IntTensor([[1, 1], [3, 3], [2, 3]])).cuda()
    labels_row_tensor = Variable(torch.IntTensor([[1, 1], [3, 3], [2, 3]]))
    warp_ctc_loss_interface = WarpCTCLossInterface.create_warp_ctc_loss_interface()

    # loss = warp_ctc_loss_interface.compute_ctc_loss_version_two(probs, labels_row_tensor)
    batch_size = 3
    loss = warp_ctc_loss_interface.compute_ctc_loss(probs, labels_row_tensor, batch_size)
    print("loss: " + str(loss))
    # loss.backward()


def main():
    test_warp_ctc_loss_interface()

    #cifar_ten_basic_recognition()


if __name__ == "__main__":
    main()
