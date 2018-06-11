from torch.utils.data import Dataset, DataLoader
from data_preprocessing.iam_database_preprocessing.iam_lines_dictionary import IamLinesDictionary
from data_preprocessing.iam_database_preprocessing.iam_lines_dictionary import IamLineInformation
from data_preprocessing.iam_database_preprocessing.string_to_index_mapping_table import StringToIndexMappingTable
from skimage import io, transform
import torch
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

        print("len(result): " + str(len(result)))
        return result

    @staticmethod
    def create_string_to_index_mapping_table(examples_line_information):
        string_to_index_mapping_table = StringToIndexMappingTable.create_string_to_index_mapping_table()
        # print("examples_line_information: \n" + str(examples_line_information))
        for iam_line_information in examples_line_information:
            letters = iam_line_information.get_letters()
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

    def __len__(self):
        return len(self.get_examples())

    def __getitem__(self, idx):
        line_information = self.examples_line_information[idx]
        print("__getitem__ line_information: " + str(line_information))
        image_file_path = self.iam_lines_dictionary.get_image_file_path(line_information)
        print("image_file_path: " + str(image_file_path))

        image = io.imread(image_file_path)
        labels = []
        # FIXME: labels should contain the labels as an ndarray of integers
        # collected using string_to_index_mapping_table
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        torch_image = torch.from_numpy(image)
        print("torch_image.size(): " + str(torch_image.size()))
        # FIXME: labels must be an np.ndarray
        return {'image': torch_image,
                'labels': torch.from_numpy(labels)}


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
    torch_tensor = to_tensor(sample)
    print("torch_tensor.size(): " + str(to_tensor.size()))


def main():
    test_iam_lines_dataset()


if __name__ == "__main__":
    main()
