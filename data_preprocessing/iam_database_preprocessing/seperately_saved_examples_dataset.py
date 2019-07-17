from torch.utils.data import Dataset
import torch
import os
import util.file_utils

class SeparatelySavedExamplesDataset(Dataset):
    """
    This class implements a Dataset (for pre-processed images) that stores and loads the
    pre-processed images from individual files. This helps to avoid out-of-memory
    problems for larger datasets.
    Based on the discussion at: https://discuss.pytorch.org/t/loading-huge-data-functionality/346/2

    """
    def __init__(self, dataset_examples_folder_path: str):
        print("Creating SeparatelySavedExamplesDataset...")
        self.dataset_examples_folder_path = dataset_examples_folder_path
        self.data_files = os.listdir(dataset_examples_folder_path)
        self.data_files.sort()

    def __getitem__(self, idx: int):
        path = self.dataset_examples_folder_path + "/" + self.data_files[idx]
        return SeparatelySavedExamplesDataset.load_example_from_file_using_file_path(path)

    def __len__(self):
        return len(self.data_files)


    @staticmethod
    def example_path(dataset_examples_folder_path, example_index: int):
        return dataset_examples_folder_path + "/" + "example_" + str(example_index) + ".pt"

    @staticmethod
    def save_example_to_file(dataset_examples_folder_path, example: tuple, example_index: int):
        example_name = SeparatelySavedExamplesDataset.example_path(dataset_examples_folder_path, example_index)
        print(">>>Saving example to \"" + example_name + "\"...")
        torch.save(example, example_name)
        print("done.")

    @staticmethod
    def load_example_from_file_using_file_path(file_path: str):
        print(
            ">>>Loading example from saved file \"" + file_path + "\"...")
        example = torch.load(file_path)
        print("done.")
        return example

    @staticmethod
    def load_example_from_file_using_example_index(dataset_examples_folder_path, example_index: int):
        example_path = SeparatelySavedExamplesDataset.example_path(
            dataset_examples_folder_path, example_index)
        print(
            ">>>Loading example from saved file \"" + example_path + "\"...")
        example = torch.load(example_path)
        print("done.")
        return example

    @staticmethod
    def nonempty_file_for_example_index_exists(
            dataset_examples_folder_path, example_index: int):
        example_path = SeparatelySavedExamplesDataset.example_path(
            dataset_examples_folder_path, example_index)
        return util.file_utils.FileUtils.file_exist_and_is_not_empty(example_path)

