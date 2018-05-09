import os.path
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import util.image_visualization
import matplotlib.pyplot as plt
import torchvision
from util.image_input_transformer import ImageInputTransformer
from random import randint


def get_train_loader(batch_size):
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 32*32
    trans = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    #trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    return train_loader


def get_item_tensors_and_labels_combined(train_set: list, start_index: int, sequence_length: int):
    print("get_item_tensors_and_labels_combined - start_index: " + str(start_index) +
          " sequence length: " + str(sequence_length))
    item_one = train_set[start_index]
    item_one_tensor = item_one[0]
    item_one_label = item_one[1]

    item_tensors_combined = item_one_tensor
    item_labels_combined = list([item_one_label])

    for j in range(start_index + 1, start_index + sequence_length):
        index = j
        print("index:  " + str(index))
        next_item = train_set[index]
        next_item_tensor = next_item[0]
        next_item_label = next_item[1]

        item_tensors_combined = torch.cat((item_tensors_combined, next_item_tensor), 2)
        item_labels_combined.append(next_item_label)

    print("item_tensors_combined.size(): " + str(item_tensors_combined.size()))
    print("item_labels_combined: " + str(item_labels_combined))
    return item_tensors_combined, item_labels_combined


# This data loader creates training examples of a specified length using
# MNIST digits by concatenating them. This data loader will be used as a toy,
# development data set for handwriting recognition of sequences
# This is a first step towards using a more complicated data loader that generates examples of
# random length, concatenation 1 to N digits for each example. This latter case is
# particularly complicated, as it requires padding to keep batches with the same size
# during training.
def get_multi_digit_train_loader_fixed_length(batch_size, sequence_length):
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 16*16
    trans = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    triples_train_set = list([])
    print("train_set: " + str(train_set))

    # To avoid trying to form more sequences than fits in the training length,
    # (sequence_length -1) must be subtracted from the length of the train_set
    # when determining the upper range
    upper_range = len(train_set) - (sequence_length - 1)

    for i in range(0, upper_range, sequence_length):
        item_tensors_combined, item_labels_combined = get_item_tensors_and_labels_combined(
            train_set, i, sequence_length)
        triples_train_set.append(tuple((item_tensors_combined, item_labels_combined)))

        # util.image_visualization.imshow(torchvision.utils.make_grid(item_tensors_combined))
        # plt.show()

    train_loader = torch.utils.data.DataLoader(
        dataset=triples_train_set,
        batch_size=batch_size,
        shuffle=True)
    return train_loader


# This data loader creates training examples with elements that are concatenated
# sequences of a random length of min_num_digits (including)
# to max_num_digits (including) elements
# https://discuss.pytorch.org/t/solved-training-lstm-by-using-samples-of-various-sequence-length/9641
def get_multi_digit_train_loader_random_length(batch_size,
                                               min_num_digits, max_num_digits):
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 16*16
    trans = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    triples_train_set = list([])
    print("train_set: " + str(train_set))
    i = 0
    while i < len(train_set):
        sequence_length = randint(min_num_digits, max_num_digits)

        if i + sequence_length - 1 >= len(train_set):
            print("Selected sequence  length " + str(sequence_length) +
                  " is no longer available in remaining items... " +
                  "skipping last items")
            break

        item_tensors_combined, item_labels_combined = get_item_tensors_and_labels_combined(
            train_set, i, sequence_length)
        triples_train_set.append(tuple((item_tensors_combined, item_labels_combined)))

        #util.image_visualization.imshow(torchvision.utils.make_grid(item_tensors_combined))
        #plt.show()
        i += sequence_length

    train_loader = torch.utils.data.DataLoader(
        dataset=triples_train_set,
        batch_size=batch_size,
        shuffle=True)
    return train_loader


def get_test_loader(batch_size):
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    # exec(open(project_root + "shared_imports.py").read())

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 32*32
    trans = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    #trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
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

