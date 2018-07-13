import torch.nn.functional
import torch


class LastMinutePadding:
    NETWORK_INTERNAL_WHITE_VALUE = 1

    def __init__(self, examples_are_pre_padded: bool,
                 height_required_per_network_output_row: int=None,
                 width_required_per_network_output_column: int=None):
        # A flag for pre-padded examples is added, so that height_required_per_network_output_row and
        #  width_required_per_network_output_column don't have to be provided for padding pre-padded examples
        self.examples_are_pre_padded = examples_are_pre_padded
        self.height_required_per_network_row = height_required_per_network_output_row
        self.width_required_per_network_output_column = width_required_per_network_output_column

    def pad_and_unsqueeze_list_of_examples(self, image_tensor_list):
        max_width = 0
        max_height = 0
        for image in image_tensor_list:
            image_height = image.size(1)
            image_width = image.size(2)
            max_height = max(max_height, image_height)
            max_width = max(max_width, image_width)
        required_height = max_height
        required_width = max_width

        if not self.examples_are_pre_padded:

            if self.height_required_per_network_row is None or self.width_required_per_network_output_column is None:
                raise RuntimeError("Error: Examples are not pre-padded, so height_required_per_network_output_row" +
                                   " and width_required_per_network_output_column must be provided")

            required_height += LastMinutePadding. \
                get_additional_amount_required_to_make_multiple_of_value(max_height,
                                                                         self.height_required_per_network_row)
            required_width += LastMinutePadding. \
                get_additional_amount_required_to_make_multiple_of_value(max_width,
                                                                     self.width_required_per_network_output_column)

        image_tensors_padded_and_unsqueezed = list([])
        total_pixels = 0
        total_real_pixels = 0
        total_padding_pixels = 0

        for image in image_tensor_list:
            image_height = image.size(1)
            image_width = image.size(2)
            columns_padding_required = required_width - image_width
            rows_padding_required = required_height - image_height
            rows_padding_required_top = int(rows_padding_required / 2)
            # Make sure no row gets lost through integer division
            rows_padding_required_bottom = rows_padding_required - rows_padding_required_top

            # print("columns_padding_required: " + str(columns_padding_required))
            # print("rows_padding_required: " + str(rows_padding_required))

            # See: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
            # pad last dimension (width) by 0, columns_padding_required
            # and one-but-last dimension (height) by 0, rows_padding_required
            # p2d = (0, columns_padding_required, rows_padding_required, 0)
            p2d = (0, columns_padding_required,
                   rows_padding_required_top,
                   rows_padding_required_bottom)
            # print("image.size(): " + str(image.size()))
            # We want to pad with the internal value for white, which is 1 !!!
            image_padded = torch.nn.functional. \
                pad(image, p2d, "constant", LastMinutePadding.NETWORK_INTERNAL_WHITE_VALUE)
            # Padded images must be unsqueezed on dimension 0 for concatenation
            # later on
            image_padded_unsqueezed = image_padded.unsqueeze(0)
            # print("image_padded.size(): " + str(image_padded.size()))
            image_tensors_padded_and_unsqueezed.append(image_padded_unsqueezed)

            real_pixels = image_height * image_width
            all_pixels = required_height * required_width
            padding_pixels = all_pixels - real_pixels

            total_real_pixels += real_pixels
            total_padding_pixels += padding_pixels
            total_pixels += all_pixels

        percentage_padding_pixels = (float(total_padding_pixels) / total_pixels) * 100
        percentage_real_pixels = (float(total_real_pixels) / total_pixels) * 100
        # print("batch-padded images height, width: " + str(required_height) + "," + str(required_width))
        # print("percentage real pixels: " + str(percentage_real_pixels))
        print("percentage padding pixels: " + str(percentage_padding_pixels))

        return image_tensors_padded_and_unsqueezed, required_width

    # Pad a list of examples (3-D tensors) based on the maximum height and with
    # within the examples and the constraint that the height and width
    # must be dividable by height_required_per_network_row and and
    # width_required_per_network_output_column respectively
    def pad_and_cat_list_of_examples(self, image_tensor_list):
        with torch.no_grad():
            # print("Performing last minute padding and concatenation of the provided examples for this batch...")
            image_tensors_padded_and_unsqueezed, required_width = self.pad_and_unsqueeze_list_of_examples(image_tensor_list)
            concatenated_padded_examples = torch.cat(image_tensors_padded_and_unsqueezed, 0)
            return concatenated_padded_examples, required_width

    @staticmethod
    def get_additional_amount_required_to_make_multiple_of_value(value, value_to_be_multiple_of: int):
        additional_amount_required = \
            value_to_be_multiple_of - (value % value_to_be_multiple_of)
        additional_amount_required = \
            additional_amount_required % value_to_be_multiple_of
        # print("get_additional_amount_required_to_make_multiple_of_value(" +
        #      str(value) + "," + str(value_to_be_multiple_of) + "): " + str(additional_amount_required))
        return additional_amount_required

