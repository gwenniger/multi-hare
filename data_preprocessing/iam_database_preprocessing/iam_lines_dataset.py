from torch.utils.data import Dataset, DataLoader
from data_preprocessing.iam_database_preprocessing.iam_lines_dictionary import IamLinesDictionary
from data_preprocessing.iam_database_preprocessing.iam_lines_dictionary import IamLineInformation
from data_preprocessing.iam_database_preprocessing.string_to_index_mapping_table import StringToIndexMappingTable
from skimage import io, transform
import torch
import numpy
import os


class IamLinesDataset(Dataset):
    EXAMPLE_TYPES_OK = "ok"
    EXAMPLE_TYPES_ERROR = "error"
    EXAMPLE_TYPES_ALL = "all"

    def __init__(self, iam_lines_dictionary: IamLinesDictionary,
                 examples_line_information: list,
                 string_to_index_mapping_table: StringToIndexMappingTable,
                 transform=None):
        self.iam_lines_dictionary = iam_lines_dictionary
        self.examples_line_information = examples_line_information
        self.string_to_index_mapping_table = string_to_index_mapping_table
        self.transform = transform

    @staticmethod
    def get_examples_line_information(iam_lines_dictionary: IamLinesDictionary, example_types: str):
        if example_types == IamLinesDataset.EXAMPLE_TYPES_OK:
            examples_key_value_enumeration = iam_lines_dictionary.get_ok_examples()
        elif example_types == IamLinesDataset.EXAMPLE_TYPES_ERROR:
            examples_key_value_enumeration = iam_lines_dictionary.get_error_examples()
        elif example_types == IamLinesDataset.EXAMPLE_TYPES_ALL:
            examples_key_value_enumeration = iam_lines_dictionary.get_all_examples()
        else:
            raise RuntimeError("Error: IamLinesDataSet.get_examples(): unrecognized examples type")

        result = list([])
        for key, value in examples_key_value_enumeration:
            result.append(value)

        return result

    @staticmethod
    def create_string_to_index_mapping_table(examples_line_information):
        string_to_index_mapping_table = StringToIndexMappingTable.create_string_to_index_mapping_table()
        # print("examples_line_information: \n" + str(examples_line_information))
        for iam_line_information in examples_line_information:
            letters = iam_line_information.get_characters()
            string_to_index_mapping_table.add_strings(letters)
        return string_to_index_mapping_table

    @staticmethod
    def create_iam_lines_dataset(iam_lines_dictionary: IamLinesDictionary,
                                 example_types: str = EXAMPLE_TYPES_OK, transformation=None):
        examples_line_information = IamLinesDataset.get_examples_line_information(iam_lines_dictionary, example_types)
        string_to_index_mapping_table = IamLinesDataset.create_string_to_index_mapping_table(examples_line_information)

        print("string_to_index_mapping_table: " + str(string_to_index_mapping_table))
        return IamLinesDataset(iam_lines_dictionary, examples_line_information, string_to_index_mapping_table,
                               transformation)

    def split_random_train_set_and_test_set(self, test_examples_fraction):
        random_permutation = numpy.random.permutation(self.__len__())
        last_train_index = int(self.__len__() * (1 - test_examples_fraction) - 1)
        # print("last_train_index: " + str(last_train_index))
        examples_line_information_train = \
            [self.examples_line_information[i] for i in random_permutation[0:last_train_index]]
        examples_line_information_test = \
            [self.examples_line_information[i] for i in random_permutation[last_train_index:]]
        train_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_train,
                        self.string_to_index_mapping_table, self.transform)
        test_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_test,
                                    self.string_to_index_mapping_table, self.transform)

        return train_set, test_set

    def get_vocabulary_list(self):
        return self.string_to_index_mapping_table.get_vocabulary_list()

    @staticmethod
    def get_labels_with_probabilities_length_and_real_sequence_length(labels_padded,
                                                                      original_image_width: int,
                                                                      labels_sequence_length: int):
        result = labels_padded.clone()
        result = torch.cat((result, torch.IntTensor([-original_image_width])), 0)
        result = torch.cat((result, torch.IntTensor([-labels_sequence_length])), 0)
        return result

    # This method converts a tensor with uint8 values in the range 0-255 to
    # a float tensor with values in the range 0-1
    @staticmethod
    def convert_unsigned_int_image_tensor_to_float_image_tensor(image_tensor):
        result = image_tensor.type(torch.FloatTensor)
        result = torch.div(result, 255)
        return result

    def get_data_loader_with_appropriate_padding(self, data_set, max_image_height,
                                                 max_image_width, max_labels_length,
                                                 batch_size: int):
        to_tensor = ToTensor()

        train_set_pairs = list([])

        # Rescale the image
        scale_reduction_factor = 2

        # Not only must images be padded to be all the same size, they
        # must also be of a size that is a multiple of block_size squared
        # FIXME: This is a quick and dirty hack to get the sizes right
        # Implement for general block sizes

        max_image_height = int(max_image_height / scale_reduction_factor)
        max_image_width = int(max_image_width / scale_reduction_factor)
        max_image_height = max_image_height + (64 - (max_image_height % 64))
        max_image_width = max_image_width + (8 - (max_image_width % 8))

        for sample in data_set:
            image_original = sample["image"]

            image_height_original, image_width_original = image_original.shape

            image_height = int(image_height_original / scale_reduction_factor)
            image_width = int(image_width_original / scale_reduction_factor)
            rescale = Rescale(tuple([image_height, image_width]))

            # Rescale works on ndarrays, not on pytorch tensors
            sample = rescale(sample)
            sample_pytorch = to_tensor(sample)
            image = sample_pytorch["image"]
            # print(">>> image size: " + str(image.size()))

            # Get the labels and labels length
            labels = sample_pytorch["labels"]
            labels_length = labels.size(0)




            # print("image_height: " + str(image_height))
            # print("image_width: " + str(image_width))
            # print("labels_length: " + str(labels_length))

            columns_padding_required = max_image_width - image_width
            rows_padding_required = max_image_height - image_height
            # print("columns_padding_required: " + str(columns_padding_required))
            # print("rows_padding_required: " + str(rows_padding_required))

            # See: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
            # pad last dimension (width) by 0, columns_padding_required
            # and one-but-last dimension (height) by 0, rows_padding_required
            p2d = (0, columns_padding_required, rows_padding_required, 0)
            image_padded = torch.nn.functional. \
                pad(image, p2d, "constant", 0)

            # Add additional bogus channel dimension, since a channel dimension is expected by downstream
            # users of this method
            # print("before: image_padded.size(): " + str(image_padded.size()))
            image_padded = image_padded.unsqueeze(0)
            # print("after padding: image_padded.size(): " + str(image_padded.size()))

            image_padded = IamLinesDataset.convert_unsigned_int_image_tensor_to_float_image_tensor(image_padded)
            # print("after padding and type conversion: image_padded: " + str(image_padded))

            digits_padding_required = max_labels_length - labels_length
            p1d = (0, digits_padding_required)  # pad last dim by 1 on end side
            labels_padded = torch.nn.functional. \
                pad(labels, p1d, "constant", -2)
            labels_padded = IamLinesDataset.\
                get_labels_with_probabilities_length_and_real_sequence_length(labels_padded, image_width,
                                                                              labels_length)

            train_set_pairs.append(tuple((image_padded, labels_padded)))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set_pairs,
            batch_size=batch_size,
            shuffle=True)
        return train_loader

    def get_random_train_set_test_set_data_loaders(self, batch_size: int,
                                                   test_examples_fraction: float):

        max_image_height, max_image_width = self.get_max_image_dimension()
        # print("max image height: " + str(max_image_height))
        # print("max image width: " + str(max_image_width))
        max_labels_length = self.get_max_labels_length()
        # print("max labels length: " + str(max_labels_length))

        train_set, test_set = self.split_random_train_set_and_test_set(test_examples_fraction)

        train_loader = self.get_data_loader_with_appropriate_padding(train_set, max_image_height, max_image_width,
                                                                     max_labels_length, batch_size)
        test_loader = self.get_data_loader_with_appropriate_padding(test_set, max_image_height, max_image_width,
                                                                    max_labels_length, batch_size)




        return train_loader, test_loader

    def __len__(self):
        return len(self.examples_line_information)

    def __getitem__(self, idx):
        line_information = self.examples_line_information[idx]
        # print("__getitem__ line_information: " + str(line_information))
        image_file_path = self.iam_lines_dictionary.get_image_file_path(line_information)
        # print("image_file_path: " + str(image_file_path))

        sample = {'image': self.get_image(idx), 'labels': self.get_labels(idx)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, index):
        line_information = self.examples_line_information[index]
        # print("__getitem__ line_information: " + str(line_information))
        image_file_path = self.iam_lines_dictionary.get_image_file_path(line_information)
        # print("image_file_path: " + str(image_file_path))

        image = io.imread(image_file_path)

        return image

    def get_labels(self, index):
        line_information = self.examples_line_information[index]
        # print("__getitem__ line_information: " + str(line_information))
        image_file_path = self.iam_lines_dictionary.get_image_file_path(line_information)
        # print("image_file_path: " + str(image_file_path))

        characters = line_information.get_characters()
        indices = self.string_to_index_mapping_table.get_indices(characters)

        indices_array = numpy.ndarray((len(indices)), buffer=numpy.array(indices), dtype=int)

        return indices_array

    def get_max_image_dimension(self):
        max_height = 0
        max_width = 0

        for index in range(0, self.__len__()):
            image = self.get_image(index)
            height, width = image.shape

            max_height = max(max_height, height)
            max_width = max(max_width, width)

        return max_height, max_width

    def get_max_labels_length(self):
        max_length = 0

        for index in range(0, self.__len__()):
            labels = self.get_labels(index)
            # labels has only one element, but it is still a tuple
            # hence this notation is needed
            length,  = labels.shape
            # print("length: " + str(length))
            max_length = max(max_length, length)

        return max_length


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        torch_image = torch.from_numpy(image)
        # print("torch_image.size(): " + str(torch_image.size()))
        # FIXME: labels must be an np.ndarray
        return {'image': torch_image,
                'labels': torch.from_numpy(labels).type(torch.IntTensor)}


def test_iam_lines_dataset():
    lines_file_path = "/datastore/data/iam-database/ascii/lines.txt"
    iam_database_line_images_root_folder_path = "/datastore/data/iam-database/lines"
    iam_lines_dicionary = IamLinesDictionary.create_iam_dictionary(lines_file_path,
                                                                   iam_database_line_images_root_folder_path)
    iam_lines_dataset = IamLinesDataset.create_iam_lines_dataset(iam_lines_dicionary, "ok", None)
    sample = iam_lines_dataset[0]
    print("iam_lines_dataset[0]: " + str(sample))
    print("(iam_lines_dataset[0])[image]: " + str(sample["image"]))
    to_tensor = ToTensor()
    sample_image = sample["image"]
    print("sample_image: " + str(sample_image))
    torch_tensors = to_tensor(sample)
    print("torch_tensor['image'].size(): " + str(torch_tensors['image'].size()))
    print("torch_tensor['labels'].size(): " + str(torch_tensors['labels'].size()))

    print("torch_tensor['image']: " + str(torch_tensors['image']))
    print("torch_tensor['labels']: " + str(torch_tensors['labels']))

    # for i in range(0, torch_tensors['image'].view(-1).size(0)):
    #    print("torch_tensor['image'][" + str(i) + "]: " + str(torch_tensors['image'].view(-1)[i]))


def main():
    test_iam_lines_dataset()


if __name__ == "__main__":
    main()
