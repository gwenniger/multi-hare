import torch
import torch.nn
import torch.nn as nn
import time
from modules.multi_dimensional_rnn import MDRNNCell
from modules.network_to_softmax_network import NetworkToSoftMaxNetwork
from modules.multi_dimensional_rnn import MultiDimensionalRNNBase
from modules.multi_dimensional_rnn import MultiDimensionalRNN
from modules.multi_dimensional_rnn import MultiDimensionalRNNToSingleClassNetwork
from modules.multi_dimensional_rnn import MultiDimensionalRNNFast
from modules.multi_dimensional_lstm import MultiDimensionalLSTM
from modules.block_multi_dimensional_lstm import BlockMultiDimensionalLSTM
from modules.block_multi_dimensional_lstm_layer_pair import BlockMultiDimensionalLSTMLayerPair
from modules.block_multi_dimensional_lstm_layer_pair_stacking import BlockMultiDimensionalLSTMLayerPairStacking
import data_preprocessing.load_mnist
import data_preprocessing.load_cifar_ten
from data_preprocessing.iam_database_preprocessing.iam_lines_dataset import IamLinesDataset
from data_preprocessing.iam_database_preprocessing.iam_lines_dictionary import IamLinesDictionary
from util.utils import Utils
from modules.size_two_dimensional import SizeTwoDimensional
from ctc_loss.warp_ctc_loss_interface import WarpCTCLossInterface
import util.timing
import data_preprocessing
import util.tensor_utils
import sys
import numpy as np
import torch.optim as optim
from modules.trainer import ModelProperties
from modules.trainer import Trainer
from modules.evaluator import Evaluator


def test_mdrnn_cell():
    print("Testing the MultDimensionalRNN Cell... ")
    mdrnn = MDRNNCell(10, 5, nonlinearity="relu")
    input = torch.randn(6, 3, 10, requires_grad=True)

    # print("Input: " + str(input))

    h1 = torch.randn(3, 5, requires_grad=True)
    h2 = torch.randn(3, 5, requires_grad=True)
    output = []

    for i in range(6):
        print("iteration: " + str(i))
        h2 = mdrnn(input[i], h1, h2)
        print("h2: " + str(h2))
        output.append(h2)

    print(str(output))


def test_mdrnn_one_image():
    image = data_preprocessing.load_mnist.get_first_image()
    multi_dimensional_rnn = MultiDimensionalRNN.create_multi_dimensional_rnn(64, nonlinearity="sigmoid")
    if MultiDimensionalRNNBase.use_cuda():
        multi_dimensional_rnn = multi_dimensional_rnn.cuda()
    multi_dimensional_rnn.forward(image)


def print_number_of_parameters(model):
    i = 0
    total_parameters = 0
    for parameter in model.parameters():
        parameters = 1
        for dim in parameter.size():
            parameters *= dim
        print("model.parameters[" + str(i) + "] size: " +
              str(parameter.size()) + ": " + str(parameters))
        total_parameters += parameters
        i += 1
    print("total parameters: " + str(total_parameters))

# Method takes a tensor of labels starting from 0 and increases
# all elements by one to get a tensor of labels starting form 1
def create_labels_starting_from_one(labels):
    y = torch.IntTensor([1])
    # Increase all labels by 1. This is because 0 is reserved for
    # the blank label in the warp_ctc_interface, so labels inside
    # this interface are expected to start from 1
    # See also: https://discuss.pytorch.org/t/adding-a-scalar/218
    #ones_with_last_two_elements_zero = y.expand_as(labels)
    ones_with_last_two_elements_zero = torch.ones(labels.size(0), labels.size(1), dtype=torch.int)
    # print("ones_with_last_two_elements_zero before: " + str(ones_with_last_two_elements_zero))
    # The last two elements are not labels but the negative probabilities_sequence
    # length and the negative labels_sequence length respectively. These last two
    # values should be left unchanged
    ones_with_last_two_elements_zero[:, ones_with_last_two_elements_zero.size(1) - 1] = 0
    ones_with_last_two_elements_zero[:, ones_with_last_two_elements_zero.size(1) - 2] = 0
    # print("ones_with_last_two_elements_zero: " + str(ones_with_last_two_elements_zero))
    labels_starting_from_one = labels + ones_with_last_two_elements_zero
    # print("labels_starting_from_one: " + str(labels_starting_from_one))

    return labels_starting_from_one


# https://stackoverflow.com/questions/45384684/replace-all-nonzero-values-by-zero-and-all-zero-values-by-a-specific-value
def replace_all_negative_values_by_zero(tensor):
    result = tensor.clone()
    result[tensor < 0] = 0
    return result


def train_mdrnn_no_ctc(train_loader, test_loader, input_channels: int, input_size: SizeTwoDimensional, hidden_states_size: int, batch_size,
                    compute_multi_directional: bool, use_dropout: bool,
                    vocab_list: list):

    criterion = nn.CrossEntropyLoss()

    # http://pytorch.org/docs/master/notes/cuda.html
    device = torch.device("cuda:0")
    # device_ids should include device!
    # device_ids lists all the gpus that may be used for parallelization
    # device is the initial device the model will be put on
    #device_ids = [0, 1]
    device_ids = [0]

    mdlstm_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 2)
    # mdlstm_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 4)
    block_strided_convolution_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 2)

    multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking. \
        create_three_layer_pair_network_linear_parameter_size_increase(input_channels, hidden_states_size,
                                                                       mdlstm_block_size,
                                                                       block_strided_convolution_block_size,
                                                                       compute_multi_directional,
                                                                       use_dropout)

    number_of_classes_excluding_blank = len(vocab_list) - 1
    # number_of_classes_excluding_blank = 10



    network = NetworkToSoftMaxNetwork.create_network_to_soft_max_network(multi_dimensional_rnn,
                                                                         input_size, number_of_classes_excluding_blank)
    if Utils.use_cuda():

        network = nn.DataParallel(network, device_ids=device_ids)

        network.to(device)
    else:
        raise RuntimeError("CUDA not available")

    # https://discuss.pytorch.org/t/register-backward-hook-on-nn-sequential/472/6

    print_number_of_parameters(multi_dimensional_rnn)

    # optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-5)
    optimizer = optim.Adam(network.parameters(), lr=0.000001, weight_decay=1e-5)

    start = time.time()

    num_gradient_corrections = 0

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        time_start = time.time()
        for i, data in enumerate(train_loader, 0):

            time_start_batch = time.time()

            # get the inputs
            inputs, labels = data

            # Hack the labels to be of the right size. These labels are not really
            # meaningful, but that's OK, since the method is mainly meant to test
            # the speed of the different steps
            labels = torch.zeros(batch_size, 81, dtype=torch.int64)
            print("labels: " + str(labels))

            print("labels: " + str(labels))

            # Increase all labels by one, since that is the format
            # expected by warp_ctc, which reserves the 0 label for blanks
            # labels = create_labels_starting_from_one(labels)

            if Utils.use_cuda():
                inputs = inputs.to(device)
                # Set requires_grad(True) directly and only for the input
                inputs.requires_grad_(True)



            # wrap them in Variable
            # labels = Variable(labels)  # Labels need no gradient apparently
            if Utils.use_cuda():
                labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            #print("inputs: " + str(inputs))


            # forward + backward + optimize
            #outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
            print("train_multi_dimensional_rnn_ctc.train_mdrnn - labels.size(): " + str(labels.size()))
            print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs.size(): " + str(inputs.size()))
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs: " + str(inputs))

            time_start_network_forward = time.time()
            outputs = network(inputs)
            print("Time used for network forward: " + str(util.timing.time_since(time_start_network_forward)))

            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            time_start_loss_computation = time.time()
            loss = criterion(outputs, labels)
            print("Time used for loss computation: " + str(util.timing.time_since(time_start_loss_computation)))

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.item()

            print("loss: " + str(loss))

            time_start_loss_backward = time.time()
            loss.backward()
            print("Time used for loss backward: " + str(util.timing.time_since(time_start_loss_backward)))

            print("Finished profiling block...")
            # print("len(prof.function_events): " + str(len(prof.function_events)))

            # print("Computing log string...")
            # log_string = prof.table(sort_by="cpu_time")
            # print("Done...")
            # print("Saving to log file...")
            # text_file = open('/home/gemaille/AI/handwriting-recognition/log_iam_test.txt', 'w')
            # text_file.write(log_string)
            # text_file.close()
            # print("Finished...")

            # Perform gradient clipping
            # made_gradient_norm_based_correction = clip_gradient_norm(multi_dimensional_rnn)

            # https://stackoverflow.com/questions/44796793/
            # difference-between-tf-clip-by-value-and-tf-clip-by-global-norm-for-rnns-and-how
            # With clip by value, loss is actually increasing...
            # clip_gradient_value(multi_dimensional_rnn)
            # if made_gradient_norm_based_correction:
            #    num_gradient_corrections += 1

            optimizer.step()

            running_loss += loss_value

            if i % 100 == 99:  # print every 100 mini-batches
                end = time.time()
                running_time = end - start
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100) +
                      " Running time: " + str(running_time))
                # print("Number of gradient norm-based corrections: " + str(num_gradient_corrections))
                running_loss = 0.0
                num_gradient_corrections = 0

        print("Time used for this batch: " + str(util.timing.time_since(time_start_batch)))

        percent = (i + 1) / float(len(train_loader))
        examples_processed = (i + 1) * batch_size
        total_examples = len(train_loader.dataset)
        print("Processed " + str(examples_processed) + " of " + str(total_examples) + " examples in this epoch")
        print(">>> Total time used during this epoch: " +
        str(util.timing.time_since_and_expected_remaining_time(time_start, percent)))

    print('Finished Training')

    # Run evaluation
    # multi_dimensional_rnn.set_training(False) # Normal case
    network.module.set_training(False)  # When using DataParallel
    # evaluate_mdrnn(test_loader, network, batch_size, device, vocab_list,
    #               width_reduction_factor)


# Get the data_height: the height of the examples obtained from train_loader
def get_data_height(train_loader):
    first_data_element = train_loader.dataset[0][0]
    # print("first_data_element: " + str(first_data_element))
    print("first_data_element.size(): " + str(first_data_element.size()))
    data_height = first_data_element.size(1)
    print(">>> data_height: " + str(data_height))
    return data_height


def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    # print('')
    # print('grad_input: ', type(grad_input))
    # print('grad_input[0]: ', type(grad_input[0]))
    # print('grad_output: ', type(grad_output))
    # print('grad_output[0]: ', type(grad_output[0]))
    # print('')
    # print('grad_input size:', grad_input[0].size())
    # print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())
    print('grad_output norm: ', grad_output[0].norm())


def create_model(data_height: int, input_channels: int, input_size: SizeTwoDimensional, hidden_states_size: int,
                 compute_multi_directional: bool, use_dropout: bool, vocab_list,
                 clamp_gradients: bool, data_set_name: str):
    # multi_dimensional_rnn = MultiDimensionalLSTM.create_multi_dimensional_lstm_fast(input_channels,
    #                                                                                 hidden_states_size,
    #                                                                                 compute_multi_directional,
    #                                                                                 use_dropout,
    #                                                                                 nonlinearity="sigmoid")

    # multi_dimensional_rnn = BlockMultiDimensionalLSTM.create_block_multi_dimensional_lstm(input_channels,
    #                                                                                       hidden_states_size,
    #                                                                                       mdlstm_block_size,
    #                                                                                       compute_multi_directional,
    #                                                                                       use_dropout,
    #                                                                                       nonlinearity="sigmoid")
    #
    # block_strided_convolution_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 4)
    # output_channels = mdlstm_block_size.width * mdlstm_block_size.height * hidden_states_size
    # multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPair.\
    #     create_block_multi_dimensional_lstm_layer_pair(input_channels, hidden_states_size,
    #                                                    output_channels, mdlstm_block_size,
    #                                                    block_strided_convolution_block_size,
    #                                                    compute_multi_directional,
    #                                                    use_dropout,
    #                                                    nonlinearity="tanh")

    # # An intermediate test case with first a layer-pair that consists of a
    # # BlockMultiDimensionalLSTM layer, followed by a BlockStructuredConvolution layer.
    # # After this comes an additional single block_strided_convolution layer as
    # # opposed to another full layer pair
    # mdlstm_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 4)
    # block_strided_convolution_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 4)
    # multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking.\
    #     create_one_layer_pair_plus_second_block_convolution_layer_network(hidden_states_size, mdlstm_block_size,
    #                                                                       block_strided_convolution_block_size)

    # # An intermediate test case with first a layer-pair that consists of a
    # # BlockMultiDimensionalLSTM layer, followed by a BlockStructuredConvolution layer.
    # # After this comes an additional single mdlstm layer as
    # # opposed to another full layer pair
    # mdlstm_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 4)
    # block_strided_convolution_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 4)
    # multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking.\
    #     create_one_layer_pair_plus_second_block_mdlstm_layer_network(hidden_states_size, mdlstm_block_size,
    #                                                                       block_strided_convolution_block_size)
    #
    mdlstm_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 2)
    # mdlstm_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 4)
    block_strided_convolution_block_size = SizeTwoDimensional.create_size_two_dimensional(4, 2)

    if data_set_name == "MNIST":
        multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking.\
            create_two_layer_pair_network(hidden_states_size, mdlstm_block_size,
                                      block_strided_convolution_block_size,
                                       clamp_gradients)
    # multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking. \
    #    create_three_layer_pair_network(hidden_states_size, mdlstm_block_size,
    #                                 block_strided_convolution_block_size)

    elif data_set_name == "IAM":
        multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking. \
            create_three_layer_pair_network_linear_parameter_size_increase(input_channels, hidden_states_size,
                                                                           mdlstm_block_size,
                                                                           block_strided_convolution_block_size,
                                                                           compute_multi_directional,
                                                                           clamp_gradients,
                                                                           use_dropout)
    else:
        raise RuntimeError("Error: unrecognized dataset name")

    # See: https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
    # multi_dimensional_rnn.register_backward_hook(printgradnorm)

    number_of_classes_excluding_blank = len(vocab_list) - 1

    network = NetworkToSoftMaxNetwork.create_network_to_soft_max_network(multi_dimensional_rnn,
                                                                         input_size, number_of_classes_excluding_blank,
                                                                         data_height, clamp_gradients)

    return network


def create_optimizer(network):
    # optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.001, momentum=0.9)

    # Adding some weight decay seems to do magic, see: http://pytorch.org/docs/master/optim.html
    # optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

    # Faster learning
    # optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.01, momentum=0.9)

    # https://github.com/SeanNaren/deepspeech.pytorch/blob/master/train.py
    ### Reducing the learning rate seems to reduce the infinite loss problem
    ### https://github.com/baidu-research/warp-ctc/issues/51

    # https://www.quora.com/
    # How-can-one-escape-from-a-plateau-in-Deep-Learning-by-using-the-momentum-+-RMSProp-method-hyper-parameters
    # Increase weight decay to  1e-4
    # optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4,
    #                      nesterov=True)

    # optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5,
    #                      nesterov=True)
    # optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5,
    #                      nesterov=True)
    # optimizer = optim.SGD(network.parameters(), lr=0.000005, momentum=0.9, weight_decay=1e-5,
    #                      nesterov=True)

    # Adam seems to be more robust against the infinite losses problem during weight
    # optimization, see:
    # https://github.com/SeanNaren/warp-ctc/issues/29
    # If the learning rate is too large, then for some reason the loss increases
    # after some epoch and then from that point onwards keeps increasing
    # But the largest learning rate that still works also seems on things like
    # the relative length of the output sequence
    # optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-5)
    # optimizer = optim.Adam(network.parameters(), lr=0.0001)

    # Initial value used by "Handwriting Recognition with Large Multimodal
    # Long-Short Term Memory Recurrent Neural Networks"
    optimizer = optim.Adam(network.parameters(), lr=0.0005)
    # optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-5)


    # optimizer = optim.Adam(network.parameters(), lr=0.000001, weight_decay=1e-5)
    return optimizer


def train_mdrnn_ctc(train_loader, validation_loader, test_loader, input_channels: int, input_size: SizeTwoDimensional, hidden_states_size: int, batch_size,
                    compute_multi_directional: bool, use_dropout: bool,
                    vocab_list: list, blank_symbol: str,
                    image_input_is_unsigned_int: bool,
                    data_set_name):

    # http://pytorch.org/docs/master/notes/cuda.html
    device = torch.device("cuda:0")
    # device_ids should include device!
    # device_ids lists all the gpus that may be used for parallelization
    # device is the initial device the model will be put on
    device_ids = [0, 1]
    # device_ids = [0]

    # See: https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
    # multi_dimensional_rnn.register_backward_hook(printgradnorm)

    data_height = get_data_height(train_loader)
    clamp_gradients = False
    network = create_model(data_height, input_channels, input_size, hidden_states_size,
                           compute_multi_directional, use_dropout, vocab_list,
                           clamp_gradients, data_set_name)

    # See: https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
    # network.register_backward_hook(printgradnorm)

    if Utils.use_cuda():
        # multi_dimensional_rnn = multi_dimensional_rnn.cuda()
        network = nn.DataParallel(network, device_ids=device_ids)

        network.to(device)
        #print("multi_dimensional_rnn.module.mdlstm_direction_one_parameters.parallel_memory_state_column_computation :"
        #      + str(multi_dimensional_rnn.module.mdlstm_direction_one_parameters.parallel_memory_state_column_computation))

        #print("multi_dimensional_rnn.module.mdlstm_direction_one_parameters."
        #      "parallel_memory_state_column_computation.parallel_convolution.bias :"
        #      + str(multi_dimensional_rnn.module.mdlstm_direction_one_parameters.
        #            parallel_memory_state_column_computation.parallel_convolution.bias))

        #print("multi_dimensional_rnn.module.mdlstm_direction_one_parameters."
        #      "parallel_hidden_state_column_computation.parallel_convolution.bias :"
        #      + str(multi_dimensional_rnn.module.mdlstm_direction_one_parameters.
        #            parallel_hidden_state_column_computation.parallel_convolution.bias))
    else:
        raise RuntimeError("CUDA not available")

    print_number_of_parameters(network)

    optimizer = create_optimizer(network)

    start = time.time()

    #ctc_loss = warpctc_pytorch.CTCLoss()
    warp_ctc_loss_interface = WarpCTCLossInterface.create_warp_ctc_loss_interface()
    # Get the width reduction factor which will be needed to compute the real widths
    # in the output from the real input width information in the warp_ctc_loss function
    width_reduction_factor = network.module.get_width_reduction_factor()

    model_properties = ModelProperties(image_input_is_unsigned_int, width_reduction_factor)
    trainer = Trainer(network, optimizer, warp_ctc_loss_interface, model_properties)

    iteration = 1
    for epoch in range(2):  # loop over the dataset multiple times

        # print("Time used for this batch: " + str(util.timing.time_since(time_start_batch)))

        trainer.train_one_epoch(train_loader, epoch, start, batch_size, device)

        # Update the iteration / minibatch number
        iteration += 1

        print("<validation evaluation epoch " + str(epoch) + " >")
        # Run evaluation
        # multi_dimensional_rnn.set_training(False) # Normal case
        network.module.set_training(False)  # When using DataParallel
        Evaluator.evaluate_mdrnn(validation_loader, network, device, vocab_list, blank_symbol,
                                 width_reduction_factor, image_input_is_unsigned_int)
        network.module.set_training(True)  # When using DataParallel
        print("</validation evaluation epoch " + str(epoch) + " >")

    print('Finished Training')

    print('Evaluation on test set...')

    print("<test evaluation epoch " + str(epoch) + " >")
    # Run evaluation
    # multi_dimensional_rnn.set_training(False) # Normal case
    network.module.set_training(False)  # When using DataParallel
    Evaluator.evaluate_mdrnn(test_loader, network, device, vocab_list, blank_symbol,
                             width_reduction_factor, image_input_is_unsigned_int)
    network.module.set_training(True)  # When using DataParallel
    print("</test evaluation epoch " + str(epoch) + " >")


def mnist_recognition_fixed_length():
    batch_size = 128
    number_of_digits_per_example = 2
    # In MNIST there are the digits 0-9, and we also add a symbol for blanks
    # This vocab_list will be used by the decoder
    vocab_list = list(['_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    train_loader = data_preprocessing.load_mnist.\
        get_multi_digit_train_loader_fixed_length(batch_size, number_of_digits_per_example)
    test_loader = data_preprocessing.load_mnist.\
        get_multi_digit_test_loader_fixed_length(batch_size, number_of_digits_per_example)

    # test_mdrnn_cell()
    #test_mdrnn()
    input_height = 16
    input_width = 16
    input_channels = 1
    hidden_states_size = 32
    # https://stackoverflow.com/questions/45027234/strange-loss-curve-while-training-lstm-with-keras
    # Possibly a batch size of 128 leads to more instability in training?
    #batch_size = 128

    compute_multi_directional = True
    # https://discuss.pytorch.org/t/dropout-changing-between-training-mode-and-eval-mode/6833
    use_dropout = False

    # Interesting link with tips on how to fix training:
    # https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
    # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873
    # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191

    input_size = SizeTwoDimensional.create_size_two_dimensional(input_height, input_width)
    #with torch.autograd.profiler.profile(use_cuda=False) as prof:
    train_mdrnn_ctc(train_loader, test_loader, input_channels, input_size, hidden_states_size, batch_size,
                    compute_multi_directional, use_dropout, vocab_list, "MNIST_FIXED_LENGTH")
    #print(prof)


def mnist_recognition_variable_length():
    batch_size = 128
    min_num_digits = 1
    max_num_digits = 3
    # In MNIST there are the digits 0-9, and we also add a symbol for blanks
    # This vocab_list will be used by the decoder
    vocab_list = list(['_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    train_loader = data_preprocessing.load_mnist.\
        get_multi_digit_train_loader_random_length(batch_size, min_num_digits, max_num_digits)
    test_loader = data_preprocessing.load_mnist.\
        get_multi_digit_test_loader_random_length(batch_size, min_num_digits, max_num_digits)

    # test_mdrnn_cell()
    #test_mdrnn()
    input_height = 16
    input_width = 16
    input_channels = 1
    hidden_states_size = 32
    # https://stackoverflow.com/questions/45027234/strange-loss-curve-while-training-lstm-with-keras
    # Possibly a batch size of 128 leads to more instability in training?
    #batch_size = 128

    compute_multi_directional = False
    # https://discuss.pytorch.org/t/dropout-changing-between-training-mode-and-eval-mode/6833
    use_dropout = False

    # TODO: Add gradient clipping? This might also make training more stable?
    # Interesting link with tips on how to fix training:
    # https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
    # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873
    # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191

    input_size = SizeTwoDimensional.create_size_two_dimensional(input_height, input_width)
    #with torch.autograd.profiler.profile(use_cuda=False) as prof:
    blank_symbol = "_"
    image_input_is_unsigned_int = False
    train_mdrnn_ctc(train_loader, test_loader, test_loader, input_channels, input_size, hidden_states_size, batch_size,
                    compute_multi_directional, use_dropout, vocab_list, blank_symbol,
                    image_input_is_unsigned_int, "MNIST")
    #print(prof)


def iam_recognition():

        batch_size = 20

        lines_file_path = "/datastore/data/iam-database/ascii/lines.txt"
        iam_database_line_images_root_folder_path = "/datastore/data/iam-database/lines"

        print("Loading IAM dataset...")
        iam_lines_dicionary = IamLinesDictionary.create_iam_dictionary(lines_file_path,
                                                                       iam_database_line_images_root_folder_path)
        iam_lines_dataset = IamLinesDataset.create_iam_lines_dataset(iam_lines_dicionary, "ok", None)


        # This vocab_list will be used by the decoder
        vocab_list = iam_lines_dataset.get_vocabulary_list()
        blank_symbol = iam_lines_dataset.get_blank_symbol()

        train_examples_fraction = 0.70
        validation_examples_fraction = 0.10
        test_examples_fraction = 0.20

        train_loader, validation_loader, test_loader = iam_lines_dataset.\
            get_random_train_set_validation_set_test_set_data_loaders(batch_size, train_examples_fraction,
                                                                      validation_examples_fraction,
                                                                      test_examples_fraction)
        print("Loading IAM dataset: DONE")

        # test_mdrnn_cell()
        #test_mdrnn()
        input_height = 16
        input_width = 16
        input_channels = 1
        # hidden_states_size = 32
        hidden_states_size = 8  # Start with a lower initial hidden states size since there are more layers
        # https://stackoverflow.com/questions/45027234/strange-loss-curve-while-training-lstm-with-keras
        # Possibly a batch size of 128 leads to more instability in training?
        #batch_size = 128

        compute_multi_directional = False
        # https://discuss.pytorch.org/t/dropout-changing-between-training-mode-and-eval-mode/6833
        use_dropout = False

        # TODO: Add gradient clipping? This might also make training more stable?
        # Interesting link with tips on how to fix training:
        # https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
        # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873
        # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191

        input_size = SizeTwoDimensional.create_size_two_dimensional(input_height, input_width)
        #with torch.autograd.profiler.profile(use_cuda=False) as prof:
        image_input_is_unsigned_int = True
        train_mdrnn_ctc(train_loader, validation_loader, test_loader, input_channels, input_size, hidden_states_size,
                        batch_size, compute_multi_directional, use_dropout, vocab_list, blank_symbol,
                        image_input_is_unsigned_int, "IAM")
        # train_mdrnn_no_ctc(train_loader, test_loader, input_channels, input_size, hidden_states_size, batch_size,
        #                 compute_multi_directional, use_dropout, vocab_list)


def main():
    # mnist_recognition_fixed_length()
    mnist_recognition_variable_length()
    # iam_recognition()
    #cifar_ten_basic_recognition()


if __name__ == "__main__":
    main()
