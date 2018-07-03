from abc import abstractmethod
import torch


class PaddingStrategy:
    """
    This strategy class will determine how the padding of the training examples is
    being done
    """

    def __init__(self, width_required_per_network_output_column):
        self.width_required_per_network_output_column = width_required_per_network_output_column

    @abstractmethod
    def get_collumns_padding_required(self, image_width, max_image_width):
        raise RuntimeError("not implemented")

    @abstractmethod
    def create_train_loader(self, train_set_pairs, batch_size):
        raise RuntimeError("not implemented")

    @staticmethod
    def get_additional_amount_required_to_make_multiple_of_value(value, value_to_be_multiple_of: int):
        additional_amount_required = \
            value_to_be_multiple_of - (value % value_to_be_multiple_of)
        additional_amount_required = \
            additional_amount_required % value_to_be_multiple_of
        # print("get_additional_amount_required_to_make_multiple_of_value(" +
        #      str(value) + "," + str(value_to_be_multiple_of) + "): " + str(additional_amount_required))
        return additional_amount_required

    @staticmethod
    def create_padding_strategy(width_required_per_network_output_column, minimize_horizontal_padding: bool):
        if minimize_horizontal_padding:
            return MinimalHorizontalPaddingStrategy(width_required_per_network_output_column)
        else:
            return FullPaddingStrategy(width_required_per_network_output_column)



class FullPaddingStrategy(PaddingStrategy):

    def __init__(self, width_required_per_network_output_column):
        super(FullPaddingStrategy, self).__init__(width_required_per_network_output_column)

    def get_collumns_padding_required(self, image_width, max_image_width):
        return max_image_width - image_width

    def create_train_loader(self, train_set_pairs, batch_size):

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set_pairs,
            batch_size=batch_size,
            shuffle=True)
        return train_loader


class MinimalHorizontalPaddingStrategy(PaddingStrategy):

    def __init__(self, width_required_per_network_output_column):
        super(MinimalHorizontalPaddingStrategy, self).__init__(width_required_per_network_output_column)

    def get_collumns_padding_required(self, image_width, max_image_width):
        return PaddingStrategy.\
            get_additional_amount_required_to_make_multiple_of_value(image_width,
                                                                     self.width_required_per_network_output_column)

    # a simple custom collate function
    @staticmethod
    def my_collate(batch):
        # print("my_collate - batch: " + str(batch))
        data = [item[0] for item in batch]

        # print("my_collate - data: " + str(data))
        target_list = [item[1].unsqueeze(0) for item in batch]
        target = torch.cat(target_list, 0)
        # print("my_collate - target: " + str(target))
        # target = torch.LongTensor(target)
        return [data, target]

    # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    def create_train_loader(self, train_set_pairs, batch_size):

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set_pairs,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MinimalHorizontalPaddingStrategy.my_collate,
            pin_memory=True)

        return train_loader