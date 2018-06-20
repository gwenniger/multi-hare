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
import ctcdecode
import util.timing
import data_preprocessing
import util.tensor_utils
import sys


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


# Note that if seq_len=0 then the result will always be the empty String
def convert_to_string(tokens, vocab, seq_len):
    # print("convert_to_string - tokens: " + str(tokens))
    # print("convert_to_string - vocab: " + str(vocab))
    # print("convert_to_string - seq_len: " + str(seq_len))
    result = ''.join([vocab[x] for x in tokens[0:seq_len]])
    # print("convert_to_string - result: " + str(result))
    return result


def convert_labels_tensor_to_string(labels: torch.Tensor, vocab_list: list, blank_symbol):

    # Check that the first label of vocab_list is indeed the blank label,
    # otherwise the results will be incorrect
    if not vocab_list[0] == blank_symbol:
        raise RuntimeError("Error: convert_labels_tensor_to_string - " +
                           "requires the first label of vocab_list (" +
                           str(vocab_list) + ") to be the blank symbol " +
                           " but was: " + str(vocab_list[0]))

    labels_as_list = labels.data.tolist()
    result = ""
    for i in range(0, labels.size(0)):
        index = labels_as_list[i]
        # print("convert_labels_tensor_to_string - index: " + str(index))
        result = result + str(vocab_list[index])
    return result


def evaluate_mdrnn(test_loader, multi_dimensional_rnn, device,
                   vocab_list: list, blank_symbol: str, horizontal_reduction_factor: int,
                   image_input_is_unsigned_int: bool):

    correct = 0
    total = 0

    for data in test_loader:
        inputs, labels = data

        if MultiDimensionalRNNBase.use_cuda():
            labels = labels.to(device)
            inputs = inputs.to(device)

        # If the image input comes in the form of unsigned ints, they need to
        # be converted to floats (after moving to GPU, i.e. directly on GPU
        # which is faster)
        if image_input_is_unsigned_int:
            inputs = IamLinesDataset.convert_unsigned_int_image_tensor_to_float_image_tensor(inputs)

        #outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
        outputs = multi_dimensional_rnn(inputs)

        probabilities_sum_to_one_dimension = 2
        # Outputs is the output of the linear layer which is the input to warp_ctc
        # But to get probabilities for the decoder, the softmax function needs to
        # be applied to the outputs
        probabilities = torch.nn.functional.\
            softmax(outputs, probabilities_sum_to_one_dimension)

        print(">>> evaluate_mdrnn  - outputs.size: " + str(outputs.size()))
        print(">>> evaluate_mdrnn  - probabilities.size: " + str(probabilities.size()))

        beam_size = 20
        print(">>> evaluate_mdrnn  - len(vocab_list): " + str(len(vocab_list)))
        decoder = ctcdecode.CTCBeamDecoder(vocab_list, beam_width=beam_size,
                                           blank_id=vocab_list.index(blank_symbol),
                                           num_processes=16)
        label_sizes = WarpCTCLossInterface.\
            create_sequence_lengths_specification_tensor_different_lengths(labels)

        sequence_lengths = WarpCTCLossInterface.create_probabilities_lengths_specification_tensor_different_lengths(
            labels, horizontal_reduction_factor, probabilities)
        # print(">>> evaluate_mdrnn  -  sequence lengths: " + str(sequence_lengths))
        # print("probabilities.data.size(): " + str(probabilities.data.size()))
        beam_results, beam_scores, timesteps, out_seq_len = \
            decoder.decode(probabilities.data, sequence_lengths)

        # print(">>> evaluate_mdrnn  - beam_results: " + str(beam_results))

        total += labels.size(0)

        for example_index in range(0, beam_results.size(0)):
            beam_results_sequence = beam_results[example_index][0]
            # print("beam_results_sequence: \"" + str(beam_results_sequence) + "\"")
            output_string = convert_to_string(beam_results_sequence,
                                              vocab_list, out_seq_len[example_index][0])
            example_labels_with_padding = labels[example_index]
            # Extract the real example labels, removing the padding labels
            reference_labels = example_labels_with_padding[0:label_sizes[example_index]]

            # print(">>> evaluate_mdrnn  - reference_labels: " + str(reference_labels))
            reference_labels_string = convert_labels_tensor_to_string(reference_labels, vocab_list, blank_symbol)

            if reference_labels_string == output_string:
                # print("Yaaaaah, got one correct!!!")
                correct += 1
                correct_string = "correct"
            else:
                correct_string = "wrong"

            print(">>> evaluate_mdrnn  - output: \"" + output_string + "\" " +
                  "\nreference: \"" + reference_labels_string + "\" --- "
                  + correct_string)



        # correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test inputs: %d %%' % (
            float(100 * correct) / total))


def clip_gradient_norm(model):
    made_gradient_norm_based_correction = False

    # What is a good max norm for clipping is an empirical question. But a norm
    # of 15 seems to work nicely for this problem.
    # In the beginning there is a lot of clipping,
    # but within an epoch, the total norm is nearly almost below 15
    # so that  clipping becomes almost unnecessary after the start.
    # This is probably what you want: avoiding instability but not also
    # clipping much more or stronger than necessary, as it slows down learning.
    # A max_norm of 10 also seems to work reasonably well, but worse than 15.
    # On person on Quora wrote
    # https://www.quora.com/How-should-I-set-the-gradient-clipping-value
    # "It’s very empirical. I usually set to 4~6.
    # In tensorflow seq2seq example, it is 5.
    # According to the original paper, the author suggests you could first print
    # out uncliped norm and setting value to 1/10 of the max value can still
    # make the model converge."
    # A max norm of 15 seems to make the learning go faster and yield almost no
    # clipping in the second epoch onwards, which seems ideal.
    max_norm = 15
    # https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
    # https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
    # norm_type is the p-norm type, a value of 2 means the eucledian norm
    # The higher the number of the norm_type, the higher the influence of the
    # outliers on the total_norm. For norm_type = 1 (= "manhattan distance")
    # all values have linear effect on the total norm.
    norm_type = 2

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/9
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm,
                                                norm_type)

    if total_norm > max_norm:
        made_gradient_norm_based_correction = True
        print("Made gradient norm based correction. total norm: " + str(total_norm))


    #
    return made_gradient_norm_based_correction


def clip_gradient_value(model):
    # Clipping the gradient value is an alternative to clipping the gradient norm,
    # and seems to be more effective
    # https://pytorch.org/docs/master/_modules/torch/nn/utils/clip_grad.html
    # https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
    # https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
    grad_clip_value_ = 100
    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value_)


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
    import torch.optim as optim

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

    width_reduction_factor = network.module.get_width_reduction_factor()

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
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())
    print('grad_output norm: ', grad_output[0].norm())


def train_mdrnn_ctc(train_loader, test_loader, input_channels: int, input_size: SizeTwoDimensional, hidden_states_size: int, batch_size,
                    compute_multi_directional: bool, use_dropout: bool,
                    vocab_list: list, blank_symbol: str,
                    image_input_is_unsigned_int: bool):
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    #multi_dimensional_rnn = MultiDimensionalRNN.create_multi_dimensional_rnn(hidden_states_size,
    #                                                                         batch_size,
    #                                                                         compute_multi_directional,
    #                                                                         nonlinearity="sigmoid")
    #multi_dimensional_rnn = MultiDimensionalRNNFast.create_multi_dimensional_rnn_fast(hidden_states_size,
    #                                                                                  batch_size,
    #                                                                                  compute_multi_directional,
    #                                                                                  use_dropout,
    #                                                                                  nonlinearity="sigmoid")

    #multi_dimensional_rnn = MultiDimensionalLSTM.create_multi_dimensional_lstm(hidden_states_size,
    #                                                                           batch_size,
    #                                                                           compute_multi_directional,
    #                                                                           use_dropout,
    #                                                                           nonlinearity="sigmoid")

    # http://pytorch.org/docs/master/notes/cuda.html
    device = torch.device("cuda:0")
    # device_ids should include device!
    # device_ids lists all the gpus that may be used for parallelization
    # device is the initial device the model will be put on
    device_ids = [0, 1]
    # device_ids = [0]

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
    clamp_gradients = True
    # multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking.\
    #     create_two_layer_pair_network(hidden_states_size, mdlstm_block_size,
    #                                   block_strided_convolution_block_size,
    #                                   clamp_gradients)
    #multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking. \
    #    create_three_layer_pair_network(hidden_states_size, mdlstm_block_size,
    #                                 block_strided_convolution_block_size)

    multi_dimensional_rnn = BlockMultiDimensionalLSTMLayerPairStacking. \
       create_three_layer_pair_network_linear_parameter_size_increase(input_channels, hidden_states_size,
                                                                      mdlstm_block_size,
                                                                      block_strided_convolution_block_size,
                                                                      compute_multi_directional,
                                                                      clamp_gradients,
                                                                      use_dropout)

    # See: https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
    multi_dimensional_rnn.register_backward_hook(printgradnorm)

    number_of_classes_excluding_blank = len(vocab_list) - 1

    data_height = get_data_height(train_loader)
    network = NetworkToSoftMaxNetwork.create_network_to_soft_max_network(multi_dimensional_rnn,
                                                                         input_size, number_of_classes_excluding_blank,
                                                                         data_height, clamp_gradients)

    # See: https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html
    # network.register_backward_hook(printgradnorm)


    #multi_dimensional_rnn = Net()

    if Utils.use_cuda():
        #multi_dimensional_rnn = multi_dimensional_rnn.cuda()

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

    print_number_of_parameters(multi_dimensional_rnn)

    #optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.001, momentum=0.9)


    # Adding some weight decay seems to do magic, see: http://pytorch.org/docs/master/optim.html
    # optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

    # Faster learning
    # optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.01, momentum=0.9)


    # https://github.com/SeanNaren/deepspeech.pytorch/blob/master/train.py
    ### Reducing the learning rate seems to reduce the infinite loss problem
    ### https://github.com/baidu-research/warp-ctc/issues/51

    optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5,
                          nesterov=True)
    #optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5,
    #                      nesterov=True)
    #optimizer = optim.SGD(network.parameters(), lr=0.000005, momentum=0.9, weight_decay=1e-5,
    #                      nesterov=True)

    # Adam seems to be more robust against the infinite losses problem during weight
    # optimization, see:
    # https://github.com/SeanNaren/warp-ctc/issues/29
    # If the learning rate is too large, then for some reason the loss increases
    # after some epoch and then from that point onwards keeps increasing
    # But the largest learning rate that still works also seems on things like
    # the relative length of the output sequence
    # optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-5)
    # optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=1e-5)
    # optimizer = optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-5)
#    optimizer = optim.Adam(network.parameters(), lr=0.000001, weight_decay=1e-5)

    start = time.time()

    num_gradient_corrections = 0

    #ctc_loss = warpctc_pytorch.CTCLoss()
    warp_ctc_loss_interface = WarpCTCLossInterface.create_warp_ctc_loss_interface()
    # Get the width reduction factor which will be needed to compute the real widths
    # in the output from the real input width information in the warp_ctc_loss function
    width_reduction_factor = network.module.get_width_reduction_factor()

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        time_start = time.time()
        for i, data in enumerate(train_loader, 0):

            time_start_batch = time.time()

            # get the inputs
            inputs, labels = data

            # The format expected by warp_ctc, which reserves the 0 label for blanks
            number_of_zeros = util.tensor_utils.TensorUtils.number_of_zeros(labels)
            # Check that labels indeed start from 1 as required
            if number_of_zeros != 0:
                raise RuntimeError("Error: labels tensor contains zeros, which is " +
                                   " not allowed, since 0 is reserved for blanks")

            if Utils.use_cuda():
                inputs = inputs.to(device)

            # If the image input comes in the form of unsigned ints, they need to
            # be converted to floats (after moving to GPU, i.e. directly on GPU
            # which is faster)
            if image_input_is_unsigned_int:
                inputs = IamLinesDataset.convert_unsigned_int_image_tensor_to_float_image_tensor(inputs)

            # Set requires_grad(True) directly and only for the input
            inputs.requires_grad_(True)



            # wrap them in Variable
            # labels = Variable(labels)  # Labels need no gradient apparently
            #if Utils.use_cuda():

            # Labels must remain on CPU for warp-ctc loss
            # labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            #print("inputs: " + str(inputs))


            # forward + backward + optimize
            #outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - labels.size(): " + str(labels.size()))
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs.size(): " + str(inputs.size()))
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs: " + str(inputs))

            time_start_network_forward = time.time()
            outputs = network(inputs)
            # print("Time used for network forward: " + str(util.timing.time_since(time_start_network_forward)))


            # print(">>> outputs.size(): " + str(outputs.size()))

            # print(">>> labels.size() : " + str(labels.size()))
            # print("labels: " + str(labels))
            #warp_ctc_loss_interface.
            #print(">>> labels_one_dimensional.size() : " + str(labels_one_dimensional.size()))
            #print("labels_one_dimensional: " + str(labels_one_dimensional))


            # print("outputs: " + str(outputs))
            # print("outputs.size(): " + str(outputs.size()))
            #print("labels: " + str(labels))
            number_of_examples = inputs.size(0)

            time_start_ctc_loss_computation = time.time()
            loss = warp_ctc_loss_interface.compute_ctc_loss(outputs,
                                                            labels,
                                                            number_of_examples,
                                                            width_reduction_factor)

            # print("Time used for ctc loss computation: " + str(util.timing.time_since(time_start_ctc_loss_computation)))


            # See: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/train.py
            # The averaging seems to help learning (but a smaller learning rate
            # might have the same effect!)
            loss = loss / inputs.size(0)  # average the loss by minibatch size

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.item()

            # print("loss: " + str(loss))
            #loss = criterion(outputs, labels)

            time_start_loss_backward = time.time()
            loss.backward()
            # print("Time used for loss backward: " + str(util.timing.time_since(time_start_loss_backward)))

            # Perform gradient clipping
            #made_gradient_norm_based_correction = clip_gradient_norm(multi_dimensional_rnn)
            #if made_gradient_norm_based_correction:
            #    num_gradient_corrections += 1

            #if not (loss_sum == inf or loss_sum == -inf):
            optimizer.step()

            # print statistics
            # print("loss.data: " + str(loss.data))
            # print("loss.data[0]: " + str(loss.data[0]))
            running_loss += loss_value
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            # See: https://stackoverflow.com/questions/5598181/python-multiple-prints-on-the-same-line
            #print(str(i)+",", end="", flush=True)
            if i % 10 == 9:  # print every 10 mini-batches
                end = time.time()
                running_time = end - start
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10) +
                      " Running time: " + str(running_time))
                print("Number of gradient norm-based corrections: " + str(num_gradient_corrections))
                running_loss = 0.0
                # num_gradient_corrections = 0

                percent = (i + 1) / float(len(train_loader))
                examples_processed = (i + 1) * batch_size
                total_examples = len(train_loader.dataset)
                print("Processed " + str(examples_processed) + " of " + str(total_examples) + " examples in this epoch")
                print(">>> Time used in current epoch: " +
                      str(util.timing.time_since_and_expected_remaining_time(time_start, percent)))
                sys.stdout.flush()

            # print("Time used for this batch: " + str(util.timing.time_since(time_start_batch)))

        print("<evaluation epoch " + str(epoch) + " >")
        # Run evaluation
        # multi_dimensional_rnn.set_training(False) # Normal case
        network.module.set_training(False)  # When using DataParallel
        evaluate_mdrnn(test_loader, network, device, vocab_list, blank_symbol,
                       width_reduction_factor, image_input_is_unsigned_int)
        network.module.set_training(True)  # When using DataParallel
        print("</evaluation epoch " + str(epoch) + " >")

    print('Finished Training')


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

    # TODO: Add gradient clipping? This might also make training more stable?
    # Interesting link with tips on how to fix training:
    # https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
    # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873
    # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191

    input_size = SizeTwoDimensional.create_size_two_dimensional(input_height, input_width)
    #with torch.autograd.profiler.profile(use_cuda=False) as prof:
    train_mdrnn_ctc(train_loader, test_loader, input_channels, input_size, hidden_states_size, batch_size,
                    compute_multi_directional, use_dropout, vocab_list)
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

    compute_multi_directional = True
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
    train_mdrnn_ctc(train_loader, test_loader, input_channels, input_size, hidden_states_size, batch_size,
                    compute_multi_directional, use_dropout, vocab_list, blank_symbol,
                    False)
    #print(prof)


def iam_recognition():

        batch_size = 16

        lines_file_path = "/datastore/data/iam-database/ascii/lines.txt"
        iam_database_line_images_root_folder_path = "/datastore/data/iam-database/lines"

        print("Loading IAM dataset...")
        iam_lines_dicionary = IamLinesDictionary.create_iam_dictionary(lines_file_path,
                                                                       iam_database_line_images_root_folder_path)
        iam_lines_dataset = IamLinesDataset.create_iam_lines_dataset(iam_lines_dicionary, "ok", None)


        # This vocab_list will be used by the decoder
        vocab_list = iam_lines_dataset.get_vocabulary_list()
        blank_symbol = iam_lines_dataset.get_blank_symbol()

        test_examples_fraction = 0.20
        train_loader, test_loader = iam_lines_dataset.\
            get_random_train_set_test_set_data_loaders(batch_size, test_examples_fraction)
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
        train_mdrnn_ctc(train_loader, test_loader, input_channels, input_size, hidden_states_size, batch_size,
                        compute_multi_directional, use_dropout, vocab_list, blank_symbol,
                        image_input_is_unsigned_int)
        # train_mdrnn_no_ctc(train_loader, test_loader, input_channels, input_size, hidden_states_size, batch_size,
        #                 compute_multi_directional, use_dropout, vocab_list)


def main():
    # mnist_recognition_fixed_length()
    # mnist_recognition_variable_length()
    iam_recognition()
    #cifar_ten_basic_recognition()


if __name__ == "__main__":
    main()
