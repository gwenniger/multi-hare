from torch.utils.data import Dataset
import torch
import os

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
        return SeparatelySavedExamplesDataset.load_example_from_file(path)

    def __len__(self):
        return len(self.data_files)


    @staticmethod
    def example_name(dataset_examples_folder_path, example_index: int):
        return dataset_examples_folder_path + "/" + "example_" + str(example_index) + ".pt"

    @staticmethod
    def save_example_to_file(dataset_examples_folder_path, example: tuple, example_index: int):
        print(">>>Saving example to \"" + SeparatelySavedExamplesDataset.example_name(dataset_examples_folder_path, example_index) + "\"...")
        torch.save(example, SeparatelySavedExamplesDataset.example_name(dataset_examples_folder_path, example_index))
        print("done.")

    @staticmethod
    def load_example_from_file(file_name: str):
        print(
            ">>>Loading example from saved file \"" + file_name + "\"...")
        example = torch.load(file_name)
        print("done.")
        return example
