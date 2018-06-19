import os.path
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import util.image_visualization
import matplotlib.pyplot as plt
import torchvision
from util.image_input_transformer import ImageInputTransformer
from random import randint
import torch.nn.functional

IMAGE_HEIGHT = 16
IMAGE_WIDTH = 16


def get_train_set():
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 32*32
    trans = transforms.Compose(
        [transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    return train_set


def get_test_set():
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    # exec(open(project_root + "shared_imports.py").read())

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 32*32
    trans = transforms.Compose(
        [transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    return test_set


# We want labels to start from 1, so we increase the original label by 1
# so that 0 can reserved for blanks
def get_item_label_with_labels_starting_from_one(item_label):
    return item_label + 1


def get_item_tensors_and_labels_combined(train_set: list, start_index: int, sequence_length: int):
    print("get_item_tensors_and_labels_combined - start_index: " + str(start_index) +
          " sequence length: " + str(sequence_length))
    item_one = train_set[start_index]
    item_one_tensor = item_one[0]
    item_one_label = get_item_label_with_labels_starting_from_one(item_one[1])

    item_tensors_combined = item_one_tensor
    item_labels_combined = torch.IntTensor([item_one_label])

    for j in range(start_index + 1, start_index + sequence_length):
        index = j
        print("index:  " + str(index))
        next_item = train_set[index]
        next_item_tensor = next_item[0]

        next_item_label = get_item_label_with_labels_starting_from_one(next_item[1])

        item_tensors_combined = torch.cat((item_tensors_combined, next_item_tensor), 2)
        item_labels_combined = torch.cat((item_labels_combined,  torch.IntTensor([next_item_label])), 0)

    print("item_tensors_combined.size(): " + str(item_tensors_combined.size()))
    print("item_labels_combined: " + str(item_labels_combined))
    return item_tensors_combined, item_labels_combined


# This data loader creates examples of a specified length using
# MNIST digits by concatenating them. This data loader will be used as a toy,
# development data set for handwriting recognition of sequences
# This is a first step towards using a more complicated data loader that generates examples of
# random length, concatenation 1 to N digits for each example. This latter case is
# particularly complicated, as it requires padding to keep batches with the same size
# during training.
def get_multi_digit_loader_fixed_length(batch_size, sequence_length,
                                        data_set):
    triples_train_set = list([])
    print("data_set: " + str(data_set))

    # To avoid trying to form more sequences than fits in the training length,
    # (sequence_length -1) must be subtracted from the length of the data_set
    # when determining the upper range
    upper_range = len(data_set) - (sequence_length - 1)

    for i in range(0, upper_range, sequence_length):
        item_tensors_combined, item_labels_combined = get_item_tensors_and_labels_combined(
            data_set, i, sequence_length)
        triples_train_set.append(tuple((item_tensors_combined, item_labels_combined)))

        # print("triples_train_set: " + str(triples_train_set))

        # util.image_visualization.imshow(torchvision.utils.make_grid(item_tensors_combined))
        # plt.show()

    train_loader = torch.utils.data.DataLoader(
        dataset=triples_train_set,
        batch_size=batch_size,
        shuffle=True)
    return train_loader


def get_item_labels_with_probabilities_length_and_real_sequence_length(item_labels_combined_padded,
                                                                       labels_sequence_length):
    result = item_labels_combined_padded.clone()
    probabilities_sequence_length = labels_sequence_length * IMAGE_WIDTH
    result = torch.cat((result, torch.IntTensor([-probabilities_sequence_length])), 0)
    result = torch.cat((result, torch.IntTensor([-labels_sequence_length])), 0)
    return result


# This data loader creates examples with elements that are concatenated
# sequences of a random length of min_num_digits (including)
# to max_num_digits (including) elements
# https://discuss.pytorch.org/t/solved-training-lstm-by-using-samples-of-various-sequence-length/9641
def get_multi_digit_loader_random_length(batch_size, min_num_digits, max_num_digits,
                                         data_set):
    train_set_pairs = list([])
    print("data_set: " + str(data_set))
    i = 0
    while i < len(data_set):
        sequence_length = randint(min_num_digits, max_num_digits)

        if i + sequence_length - 1 >= len(data_set):
            print("Selected sequence  length " + str(sequence_length) +
                  " is no longer available in remaining items... " +
                  "skipping last items")
            break

        item_tensors_combined, item_labels_combined = get_item_tensors_and_labels_combined(
            data_set, i, sequence_length)
        print("item_tensors_combined.size(): " + str(item_tensors_combined.size()))
        print("item_labels_combined.size(): " + str(item_labels_combined.size()))

        # https://pytorch.org/docs/master/nn.html#torch.nn.functional.pad
        digits_padding_required = max_num_digits - sequence_length
        columns_padding_required = digits_padding_required * IMAGE_WIDTH

        p1d = (0, columns_padding_required)  # pad last dim by 1 on end side
        item_tensors_combined_padded = torch.nn.functional.\
            pad(item_tensors_combined, p1d, "constant", 0)
        p1d = (0, digits_padding_required)  # pad last dim by 1 on end side
        item_labels_combined_padded = torch.nn.functional.\
            pad(item_labels_combined, p1d, "constant", -2)
        # Add the (negative) probabilities length at the end of the padded labels, so it can later
        # be retrieved as the last item
        item_labels_combined_padded = \
            get_item_labels_with_probabilities_length_and_real_sequence_length(item_labels_combined_padded,
                                                                                sequence_length)
        print("item_tensors_combined_padded.size(): " + str(item_tensors_combined_padded.size()))
        print("item_labels_combined_padded.size(): " + str(item_labels_combined_padded.size()))
        print("item_labels_combined_padded: " + str(item_labels_combined_padded))

        train_set_pairs.append(tuple((item_tensors_combined_padded, item_labels_combined_padded)))

        #util.image_visualization.imshow(torchvision.utils.make_grid(item_tensors_combined))
        #plt.show()
        i += sequence_length

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set_pairs,
        batch_size=batch_size,
        shuffle=True)
    return train_loader


def get_multi_digit_train_loader_fixed_length(batch_size, sequence_length):
    return get_multi_digit_loader_fixed_length(batch_size, sequence_length,
                                               get_train_set())


def get_multi_digit_test_loader_fixed_length(batch_size, sequence_length):
    return get_multi_digit_loader_fixed_length(batch_size, sequence_length,
                                               get_test_set())


def get_multi_digit_train_loader_random_length(batch_size, min_num_digits, max_num_digits):
    return get_multi_digit_loader_random_length(batch_size, min_num_digits, max_num_digits,
                                                get_train_set())


def get_multi_digit_test_loader_random_length(batch_size, min_num_digits, max_num_digits):
    return get_multi_digit_loader_random_length(batch_size, min_num_digits, max_num_digits,
                                                get_test_set())


def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(
        dataset=get_train_set(),
        batch_size=batch_size,
        shuffle=True)
    return train_loader


def get_test_loader(batch_size):

    test_loader = torch.utils.data.DataLoader(
        dataset=get_test_set(),
        batch_size=batch_size,
        shuffle=False)
    return test_loader


def get_first_image():

    train_loader = get_train_loader()

    print('==>>> total training batch number: {}'.format(len(train_loader)))

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    #util.image_visualization.show_random_batch_from_image_loader_with_class_labels(train_loader, classes)

    images_iteration = iter(train_loader)
    images, labels = images_iteration.next()
    # Show the first image
    image = images[0]
    print("image: " + str(image))
    #util.image_visualization.imshow(torchvision.utils.make_grid(image))
    #plt.show()
    transformed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(image)
    print("transformed_image: " + str(transformed_image))
    util.image_visualization.imshow(torchvision.utils.make_grid(transformed_image))
    plt.show()

    return image


def main():
    # get_multi_digit_train_loader_fixed_length(10, 9)
    min_num_digits = 2
    max_num_digits = 4
    get_multi_digit_train_loader_random_length(10,
                                               min_num_digits, max_num_digits)


if __name__ == "__main__":
    main()

