import copy
from abc import abstractmethod
from abc import ABC
import torch
from data_preprocessing.last_minute_padding import LastMinutePadding

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


NUM_WORKERS = 8
#NUM_WORKERS = 2
#NUM_WORKERS = 0

class PaddingStrategy(ABC):
    """
    This strategy class will determine how the padding of the training examples is
    being done
    """

    def __init__(self, height_required_per_network_row: int,
                 width_required_per_network_output_column: int):
        self.height_required_per_network_row = height_required_per_network_row
        self.width_required_per_network_output_column = width_required_per_network_output_column

    @abstractmethod
    def get_rows_padding_required(self, image_width, max_image_width):
        raise RuntimeError("not implemented")

    @abstractmethod
    def get_columns_padding_required(self, image_width, max_image_width):
        raise RuntimeError("not implemented")

    @abstractmethod
    def create_data_loader(self, train_set_pairs, batch_size,
                           shuffle: bool):
        raise RuntimeError("not implemented")

    @staticmethod
    def create_padding_strategy(height_required_per_network_row: int,
                                width_required_per_network_output_column: int,
                                minimize_vertical_padding: bool,
                                minimize_horizontal_padding: bool,
                                perform_horizontal_batch_padding_in_data_loader):
        if minimize_horizontal_padding:
            if minimize_vertical_padding:

                if perform_horizontal_batch_padding_in_data_loader:
                    print("Performing horizontal batch-padding in dataloader...")

                return MinimalHorizontalAndVerticalPaddingStrategy(height_required_per_network_row,
                                                                   width_required_per_network_output_column,
                                                                   perform_horizontal_batch_padding_in_data_loader)
            else:
                if perform_horizontal_batch_padding_in_data_loader:
                    print("Performing horizontal batch-padding in dataloader...")

                return MinimalHorizontalPaddingStrategy(height_required_per_network_row,
                                                        width_required_per_network_output_column,
                                                        perform_horizontal_batch_padding_in_data_loader)
        else:
            return FullPaddingStrategy(height_required_per_network_row,
                                       width_required_per_network_output_column)


class FullPaddingStrategy(PaddingStrategy):

    def __init__(self, height_required_per_network_row,
                 width_required_per_network_output_column):
        super(FullPaddingStrategy, self).__init__(height_required_per_network_row,
                                                  width_required_per_network_output_column)

    def get_rows_padding_required(self, image_height, max_image_height):
        return max_image_height - image_height

    def get_columns_padding_required(self, image_width, max_image_width):
        return max_image_width - image_width

    def create_data_loader(self, train_set_pairs, batch_size,
                           shuffle: bool):

        print(">>>Creating dataloader using full padding strategy.")

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set_pairs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS)

        raise RuntimeError("Here!")

        return train_loader


class MinimalHorizontalPaddingStrategyBase(PaddingStrategy):
    EXAMPLES_ARE_PRE_PADDED = True
    # Last minute padding that will exploit the fact that the examples are already pre-padded, to avoid
    # having to provide height_required_per_network_output_row and width_required_per_network_output_column
    # as arguments
    LAST_MINUTE_PADDING = LastMinutePadding(EXAMPLES_ARE_PRE_PADDED)

    # print("my_collate - batch: " + str(batch))


    def __init__(self, height_required_per_network_row,
                 width_required_per_network_output_column,
                 perform_horizontal_batch_padding_in_data_loader: bool
                 ):
        super(MinimalHorizontalPaddingStrategyBase, self).__init__(height_required_per_network_row,
                                                                   width_required_per_network_output_column)
        self.perform_horizontal_batch_padding_in_data_loader = perform_horizontal_batch_padding_in_data_loader

    @abstractmethod
    def get_rows_padding_required(self, image_width, max_image_width):
        raise RuntimeError("not implemented")

    def get_columns_padding_required(self, image_width, max_image_width):
        return LastMinutePadding.\
            get_additional_amount_required_to_make_multiple_of_value(image_width,
                                                                     self.width_required_per_network_output_column)

    """    
    A simple custom collate function.
    This collate function keeps the original batch of examples 
    and adds them to a list. This means the user of the collate 
    function will have to deal with processing this list of 
    unequal size examples
    """
    @staticmethod
    def simple_collate_no_data_padding(batch):
        # print(">>>simple_collate_no_data_padding")
        # # print("my_collate - batch: " + str(batch))
        # data = [copy.deepcopy(item[0]) for item in batch]
        #
        # # print("my_collate - data: " + str(data))
        # target_list = [copy.deepcopy(item[1].unsqueeze(0)) for item in batch]
        # target = torch.cat(target_list, 0)
        # # print("my_collate - target: " + str(target))
        # # target = torch.LongTensor(target)
        # return [data, target]



        #### More optimized implementation ###
        # 1. Efficiently unpack the list of (data, target) tuples
        data, labels_lists = zip(*batch)

        # 2. Convert the tuple of labels_lists into a single batch tensor
        labels_tensor = torch.stack(labels_lists, dim=0)

        #return [list(data), target]
        return [data, labels_tensor]


        # a simple custom collate function

    # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    def create_data_loader(self, train_set_pairs, batch_size,
                           shuffle: bool):

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set_pairs,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.get_collate_function(),
            pin_memory=False,
            num_workers=NUM_WORKERS)

        return train_loader

    def get_collate_function(self):
        if self.perform_horizontal_batch_padding_in_data_loader:
            print(">>>Use \"collate_horizontal_last_minute_data_padding\" collate function ")
            return MinimalHorizontalPaddingStrategy.collate_horizontal_last_minute_data_padding
        else:
            print(">>>Use \"simple_collate_no_data_padding\" collate function ")
            return MinimalHorizontalPaddingStrategyBase.simple_collate_no_data_padding

    @staticmethod
    def collate_horizontal_last_minute_data_padding(batch):
        print(">>>collate_horizontal_last_minute_data_padding")

        data = [item[0] for item in batch]

        data_padded_and_concatenated, required_width = (
            MinimalHorizontalPaddingStrategyBase.LAST_MINUTE_PADDING.pad_and_cat_list_of_examples(data))

        # print("my_collate - data: " + str(data))
        target_list = [item[1].unsqueeze(0) for item in batch]
        target = torch.cat(target_list, 0)
        # print("my_collate - target: " + str(target))
        # target = torch.LongTensor(target)
        return [data_padded_and_concatenated, target]


class MinimalHorizontalPaddingStrategy(MinimalHorizontalPaddingStrategyBase):

    def __init__(self, height_required_per_network_row,
                 width_required_per_network_output_column,
                 perform_horizontal_batch_padding_in_data_loader: bool):
        super(MinimalHorizontalPaddingStrategyBase, self).__init__(height_required_per_network_row,
                                                                   width_required_per_network_output_column,
                                                                   perform_horizontal_batch_padding_in_data_loader)

    def get_rows_padding_required(self, image_height, max_image_height):
        return max_image_height - image_height


class MinimalHorizontalAndVerticalPaddingStrategy(MinimalHorizontalPaddingStrategyBase):

    def __init__(self, height_required_per_network_row,
                 width_required_per_network_output_column,
                 perform_horizontal_batch_padding_in_data_loader: bool):
        super(MinimalHorizontalAndVerticalPaddingStrategy, self).\
            __init__(height_required_per_network_row,
                     width_required_per_network_output_column,
                     perform_horizontal_batch_padding_in_data_loader)

    def get_rows_padding_required(self, image_height, max_image_height):
        return LastMinutePadding.\
            get_additional_amount_required_to_make_multiple_of_value(image_height,
                                                                 self.height_required_per_network_row)

