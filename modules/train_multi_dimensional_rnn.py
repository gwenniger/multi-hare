import torch
import torch.nn
import torch.nn as nn
import time
from modules.multi_dimensional_rnn import MDRNNCell
from modules.multi_dimensional_rnn import MultiDimensionalRNNBase
from modules.multi_dimensional_rnn import MultiDimensionalRNN
from modules.multi_dimensional_rnn import MultiDimensionalRNNFast
from modules.multi_dimensional_lstm import MultiDimensionalLSTM
import data_preprocessing.load_mnist
from util.utils import Utils


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


def evaluate_mdrnn(multi_dimensional_rnn, batch_size):
    device = torch.device("cuda:0")

    correct = 0
    total = 0
    test_loader = data_preprocessing.load_mnist.get_test_loader(batch_size)
    for data in test_loader:
        images, labels = data

        if MultiDimensionalRNNBase.use_cuda():
            labels = labels.to(device)
            images = images.to(device)

        #outputs = multi_dimensional_rnn(Variable(images))  # For "Net" (Le Net)
        outputs = multi_dimensional_rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def clip_gradient(model):
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
    max_norm = 30
    # https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
    # https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
    # grad_clip_value_ = 1
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

    # Clipping the gradient value is an alternative to clipping the gradient norm,
    # and seems to be more effective
    # https://pytorch.org/docs/master/_modules/torch/nn/utils/clip_grad.html
    # torch.nn.utils.clip_grad_value_(multi_dimensional_rnn.parameters(), grad_clip_value_)
    #
    return made_gradient_norm_based_correction


def train_mdrnn(input_height: int, input_width: int,
                hidden_states_size: int, batch_size,  compute_multi_directional: bool, use_dropout: bool):
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    #multi_dimensional_rnn = MultiDimensionalRNN.create_multi_dimensional_rnn(hidden_states_size,
    #                                                                         batch_size,
    #                                                                         compute_multi_directional,
    #                                                                         nonlinearity="sigmoid")
    #multi_dimensional_rnn = MultiDimensionalRNNFast.create_multi_dimensional_rnn_fast(input_height, input_width,
    #                                                                                  hidden_states_size,
    #                                                                                  batch_size,
    #                                                                                  compute_multi_directional,
    #                                                                                  use_dropout,
    #                                                                                   nonlinearity="sigmoid")

    #multi_dimensional_rnn = MultiDimensionalLSTM.create_multi_dimensional_lstm(hidden_states_size,
    #                                                                           batch_size,
    #                                                                           compute_multi_directional,
    #                                                                           use_dropout,
    #                                                                           nonlinearity="sigmoid")

    # http://pytorch.org/docs/master/notes/cuda.html
    device = torch.device("cuda:0")

    multi_dimensional_rnn = MultiDimensionalLSTM.create_multi_dimensional_lstm_fast(input_height,
                                                                                    input_width,
                                                                                    hidden_states_size,
                                                                                    batch_size,
                                                                                    compute_multi_directional,
                                                                                    use_dropout,
                                                                                    nonlinearity="sigmoid")

    #multi_dimensional_rnn = Net()

    if Utils.use_cuda():
        #multi_dimensional_rnn = multi_dimensional_rnn.cuda()

        multi_dimensional_rnn = nn.DataParallel(multi_dimensional_rnn, device_ids=[0])
        multi_dimensional_rnn.to(device)
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

    print_number_of_parameters(multi_dimensional_rnn)

    #optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.001, momentum=0.9)


    # Adding some weight decay seems to do magic, see: http://pytorch.org/docs/master/optim.html
    optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

    # Faster learning
    #optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.01, momentum=0.9)

    trainloader = data_preprocessing.load_mnist.get_train_loader(batch_size)

    start = time.time()

    num_gradient_corrections = 0

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            if Utils.use_cuda():
                inputs = inputs.to(device)


            # wrap them in Variable
            # labels = Variable(labels)  # Labels need no gradient apparently
            if Utils.use_cuda():
                labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            #print("inputs: " + str(inputs))


            # forward + backward + optimize
            #outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
            outputs = multi_dimensional_rnn(inputs)
            # print("outputs: " + str(outputs))
            #print("labels: " + str(labels))
            loss = criterion(outputs, labels)
            loss.backward()

            # Perform gradient clipping
            made_gradient_norm_based_correction = clip_gradient(multi_dimensional_rnn)
            if made_gradient_norm_based_correction:
                num_gradient_corrections += 1

            optimizer.step()

            # print statistics
            # print("loss.data: " + str(loss.data))
            # print("loss.data[0]: " + str(loss.data[0]))
            running_loss += loss.data
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            # See: https://stackoverflow.com/questions/5598181/python-multiple-prints-on-the-same-line
            #print(str(i)+",", end="", flush=True)
            if i % 100 == 99:  # print every 100 mini-batches
                end = time.time()
                running_time = end - start
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100) +
                      " Running time: " + str(running_time))
                print("Number of gradient norm-based corrections: " + str(num_gradient_corrections))
                running_loss = 0.0
                num_gradient_corrections = 0

    print('Finished Training')

    # Run evaluation
    # multi_dimensional_rnn.set_training(False) # Normal case
    multi_dimensional_rnn.module.set_training(False)  # When using DataParallel
    evaluate_mdrnn(multi_dimensional_rnn, batch_size)


def main():
    # test_mdrnn_cell()
    #test_mdrnn()
    input_height = 16
    input_width = 16
    hidden_states_size = 32
    # https://stackoverflow.com/questions/45027234/strange-loss-curve-while-training-lstm-with-keras
    # Possibly a batch size of 128 leads to more instability in training?
    #batch_size = 128
    batch_size = 256
    compute_multi_directional = False
    # https://discuss.pytorch.org/t/dropout-changing-between-training-mode-and-eval-mode/6833
    use_dropout = False

    # TODO: Add gradient clipping? This might also make training more stable?
    # Interesting link with tips on how to fix training:
    # https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
    # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873
    # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191

    #with torch.autograd.profiler.profile(use_cuda=False) as prof:
    train_mdrnn(input_height, input_width,
                hidden_states_size, batch_size,  compute_multi_directional, use_dropout)
    #print(prof)


if __name__ == "__main__":
    main()
