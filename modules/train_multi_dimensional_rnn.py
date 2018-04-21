import torch
from torch.autograd import Variable
import torch.nn
import torch.nn as nn
import time

from modules.multi_dimensional_rnn import MDRNNCell
from modules.multi_dimensional_rnn import MultiDimensionalRNNBase
from modules.multi_dimensional_rnn import MultiDimensionalRNN
from modules.multi_dimensional_lstm import MultiDimensionalLSTM
import data_preprocessing.load_mnist


def test_mdrnn_cell():
    print("Testing the MultDimensionalRNN Cell... ")
    mdrnn = MDRNNCell(10, 5, nonlinearity="relu")
    input = Variable(torch.randn(6, 3, 10))

    # print("Input: " + str(input))

    h1 = Variable(torch.randn(3, 5))
    h2 = Variable(torch.randn(3, 5))
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


def evaluate_mdrnn(multi_dimensional_rnn, batch_size):
    correct = 0
    total = 0
    test_loader = data_preprocessing.load_mnist.get_test_loader(batch_size)
    for data in test_loader:
        images, labels = data

        if MultiDimensionalRNNBase.use_cuda():
            labels = labels.cuda()

        #outputs = multi_dimensional_rnn(Variable(images))  # For "Net" (Le Net)
        outputs = multi_dimensional_rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def train_mdrnn(batch_size, compute_multi_directional: bool):
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    #multi_dimensional_rnn = MultiDimensionalRNN.create_multi_dimensional_rnn(batch_size,
    #                                                                         compute_multi_directional,
    #                                                                         nonlinearity="sigmoid",
    #                                                                         )
    multi_dimensional_rnn = MultiDimensionalLSTM.create_multi_dimensional_lstm(batch_size,
                                                                               compute_multi_directional,
                                                                               nonlinearity="sigmoid",
                                                                              )
    #multi_dimensional_rnn = Net()

    if MultiDimensionalRNNBase.use_cuda():
        multi_dimensional_rnn = multi_dimensional_rnn.cuda()


    #optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.001, momentum=0.9)

    # Faster learning
    optimizer = optim.SGD(multi_dimensional_rnn.parameters(), lr=0.01, momentum=0.9)

    trainloader = data_preprocessing.load_mnist.get_train_loader(batch_size)

    start = time.time()

    for epoch in range(4):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # See: https://stackoverflow.com/questions/48015235/i-get-this-error-on-pytorch-runtimeerror-invalid-argument-2-size-1-x-400?rq=1
            #LRTrans = transforms.Compose(
            #    [transforms.Scale((32, 32)),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            #inputs = LRTrans(inputs)

            # wrap them in Variable
            labels = Variable(labels)
            if MultiDimensionalRNNBase.use_cuda():
                labels = labels.cuda()

            #labels, inputs = Variable(labels), Variable(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            #print("inputs: " + str(inputs))


            # forward + backward + optimize
            #outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
            outputs = multi_dimensional_rnn(inputs)
            #print("outputs: " + str(outputs))
            #print("labels: " + str(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            if i % 200 == 199:  # print every 200 mini-batches
                end = time.time()
                running_time = end - start
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000) +
                      " Running time: " + str(running_time))
                running_loss = 0.0

    print('Finished Training')

    # Run evaluation
    evaluate_mdrnn(multi_dimensional_rnn, batch_size)

def main():
    # test_mdrnn_cell()
    #test_mdrnn()
    batch_size = 128
    compute_multi_directional = False
    train_mdrnn(batch_size, compute_multi_directional)


if __name__ == "__main__":
    main()
