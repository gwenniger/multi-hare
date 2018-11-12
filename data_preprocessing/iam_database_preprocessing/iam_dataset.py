from torch.utils.data import Dataset
from data_preprocessing.iam_database_preprocessing.iam_examples_dictionary import IamExamplesDictionary
from data_preprocessing.iam_database_preprocessing.string_to_index_mapping_table import StringToIndexMappingTable
from skimage import io, transform
import torch
import numpy
import os
import matplotlib.pyplot as plt
import sys
import util.image_visualization
from data_preprocessing.iam_database_preprocessing.data_permutation import DataPermutation
import os.path
from data_preprocessing.padding_strategy import PaddingStrategy
from data_preprocessing.last_minute_padding import LastMinutePadding
from util.tensor_block_stacking import TensorBlockStacking
from modules.size_two_dimensional import SizeTwoDimensional


class IamLinesDataset(Dataset):
    EXAMPLE_TYPES_OK = "ok"
    # "error stands for error in segmentation. But notice what it says in the
    # iam file lines.txt: "
    # #  "notice: if the line could not be properly segmented
    # #                                the transcription and extraction of the whole
    # #                                line should not be affected negatively "
    # in other words, there is no argument against using these lines, and they
    # are actually normally used in the literature.
    EXAMPLE_TYPES_ERROR_IN_WORD_SEGMENTATION = "error"
    EXAMPLE_TYPES_ALL = "all"
    UINT8_WHITE_VALUE = 255

    # Fractions examples train, validation and test set. Hard-coded for now
    TRAIN_EXAMPLES_FRACTION = 0.80
    VALIDATION_EXAMPLES_FRACTION = 0.10
    TEST_EXAMPLES_FRACTION = 0.10

    def __init__(self, iam_lines_dictionary: IamExamplesDictionary,
                 examples_line_information: list,
                 string_to_index_mapping_table: StringToIndexMappingTable,
                 height_required_per_network_output_row: int,
                 width_required_per_network_output_column: int,
                 transform=None):
        self.iam_lines_dictionary = iam_lines_dictionary
        self.examples_line_information = examples_line_information
        self.string_to_index_mapping_table = string_to_index_mapping_table
        self.transform = transform
        self.height_required_per_network_output_row = height_required_per_network_output_row
        self.width_required_per_network_output_column = width_required_per_network_output_column

    @staticmethod
    def get_examples_line_information(iam_lines_dictionary: IamExamplesDictionary, example_types: str):
        if example_types == IamLinesDataset.EXAMPLE_TYPES_OK:
            examples_key_value_enumeration = iam_lines_dictionary.get_ok_examples()
        elif example_types == IamLinesDataset.EXAMPLE_TYPES_ERROR_IN_WORD_SEGMENTATION:
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
    def create_or_load_string_to_index_mapping_table(examples_line_information,
                                                     string_to_index_table_save_or_load_path: str):

        if os.path.isfile(string_to_index_table_save_or_load_path):
            return StringToIndexMappingTable.\
                read_string_to_index_mapping_table_from_file(string_to_index_table_save_or_load_path)

        string_to_index_mapping_table = StringToIndexMappingTable.create_string_to_index_mapping_table()
        # print("examples_line_information: \n" + str(examples_line_information))
        for iam_line_information in examples_line_information:
            letters = iam_line_information.get_characters_with_word_separator()
            string_to_index_mapping_table.add_strings(letters)

        string_to_index_mapping_table.\
            save_string_to_index_mapping_table_to_file(string_to_index_table_save_or_load_path)
        return string_to_index_mapping_table

    @staticmethod
    def create_iam_dataset(iam_lines_dictionary: IamExamplesDictionary,
                           save_word_to_string_mapping_table_path: str,
                           example_types: str = EXAMPLE_TYPES_ALL,
                           transformation=None
                           ):
        examples_line_information = IamLinesDataset.get_examples_line_information(iam_lines_dictionary, example_types)

        string_to_index_mapping_table = IamLinesDataset.\
            create_or_load_string_to_index_mapping_table(examples_line_information,
                                                         save_word_to_string_mapping_table_path)

        # TODO : compute these from an input parameter
        height_required_per_network_output_row = 64
        width_required_per_network_output_column = 8

        print("string_to_index_mapping_table: " + str(string_to_index_mapping_table))
        return IamLinesDataset(iam_lines_dictionary, examples_line_information, string_to_index_mapping_table,
                               height_required_per_network_output_row, width_required_per_network_output_column,
                               transformation)

    @staticmethod
    def create_iam_words_dataset_from_input_files(
            iam_database_lines_file_path: str, iam_database_line_images_root_folder_path: str,
                vocabulary_file_path: str, example_types: str = EXAMPLE_TYPES_ALL, transformation=None):
        iam_words_dicionary = IamExamplesDictionary. \
            create_iam_words_dictionary(iam_database_lines_file_path,
                                        iam_database_line_images_root_folder_path,
                                        False)
        iam_words_dataset = IamLinesDataset.create_iam_dataset(iam_words_dicionary,
                                                               vocabulary_file_path,
                                                               example_types, transformation)
        return iam_words_dataset

    @staticmethod
    def create_iam_lines_dataset_from_input_files(
            iam_database_lines_file_path: str, iam_database_line_images_root_folder_path: str,
            vocabulary_file_path: str, example_types: str = EXAMPLE_TYPES_ALL,
            transformation=None):
        print("Loading IAM dataset...")
        iam_lines_dicionary = IamExamplesDictionary. \
            create_iam_lines_dictionary(iam_database_lines_file_path,
                                        iam_database_line_images_root_folder_path, True)
        iam_lines_dataset = IamLinesDataset.create_iam_dataset(iam_lines_dicionary,
                                                               vocabulary_file_path,
                                                               example_types, transformation)
        return iam_lines_dataset

    def split_random_train_set_validation_set_and_test_set(self,
                                                           train_examples_fraction,
                                                           validation_examples_fraction,
                                                           permutation_save_or_load_file_path: str):

        print("Entered split_random_train_set_validation_set_and_test_set...")

        permutation_length = self.__len__()
        data_permutation = DataPermutation.load_or_create_and_save_permutation(permutation_length,
                                                                               permutation_save_or_load_file_path)
        permutation = data_permutation.permutation
        last_train_index = int(self.__len__() * train_examples_fraction - 1)
        last_validation_index = int(self.__len__() * (train_examples_fraction
                                                      + validation_examples_fraction) - 1)

        # print("last_train_index: " + str(last_train_index))
        examples_line_information_train = \
            [self.examples_line_information[i] for i in permutation[0:last_train_index]]
        examples_line_information_validation = \
            [self.examples_line_information[i] for i in permutation[last_train_index:last_validation_index]]
        examples_line_information_test = \
            [self.examples_line_information[i] for i in permutation[last_validation_index:]]
        train_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_train,
                                    self.string_to_index_mapping_table, 64, 8, self.transform)
        validation_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_validation,
                                         self.string_to_index_mapping_table, 64, 8, self.transform)
        test_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_test,
                                   self.string_to_index_mapping_table, 64, 8, self.transform)

        return train_set, validation_set, test_set

    def get_vocabulary_list(self):
        return self.string_to_index_mapping_table.get_vocabulary_list()

    def get_blank_symbol(self):
        return self.string_to_index_mapping_table.get_blank_symbol()

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
        # Necessary to put the result on the right device (GPU), otherwise
        # will be on CPU
        result = result.to(image_tensor.device)
        result = torch.div(result, 255)
        return result

    # This method converts a tensor with uint8 values in the range 0-255 to
    # a float tensor with values in the range 0-1
    @staticmethod
    def convert_unsigned_int_image_tensor_or_list_to_float_image_tensor_or_list(image_tensor):
        with torch.no_grad():

            if not isinstance(image_tensor, (list, tuple)):
                return IamLinesDataset.convert_unsigned_int_image_tensor_to_float_image_tensor(image_tensor)
            else:
                result = list([])
                for element in image_tensor:
                    # print("convert_unsigned_int_image_tensor_or_list_to_float_image_tensor_or_list" +
                    #      " element.device: " + str(element.device))
                    element_converted = IamLinesDataset.convert_unsigned_int_image_tensor_to_float_image_tensor(element)
                    result.append(element_converted)
            return result

    @staticmethod
    def compute_max_adjusted_by_scale_reduction_factor(max_value, scale_reduction_factor):
        return int((max_value + (max_value % scale_reduction_factor)) / scale_reduction_factor)

    """
        Find the smallest scale reduction factor larger or equal than min_scaling_factor, that makes
        the rescaled max_image_height fit in a multiple of self.height_required_per_network_output_row
        If less than half a row of self.height_required_per_network_output_row can be saved
        by increasing the scale reduction factor further, the value of 
        min_scaling_factor is returned 
        :return scaling factor: float
        found_suitable_higher_scale_reduction_factor: bool 
    """
    def get_scale_reduction_factor_that_minimizes_horizontal_padding(self, max_image_height, min_scaling_factor):
        height_after_min_scaling = max_image_height / min_scaling_factor
        actual_number_of_rows = float(height_after_min_scaling) / self.height_required_per_network_output_row
        print("actual number of rows: " + str(actual_number_of_rows))

        closest_number_of_rows = round(actual_number_of_rows)
        print("closest lower number of rows: " + str(closest_number_of_rows))
        # Don't increase the scale reduction factor to save less than half the value
        # of self.height_required_per_network_output_row in terms of padding
        if (actual_number_of_rows - round(actual_number_of_rows)) < 0.5:
            print("Minimal scaling factor requires less than half a row of extra padding, " +
                  "therefore not increasing the scale reduction factor further")
            return min_scaling_factor, False

        closest_lower_number_of_rows = int(height_after_min_scaling / self.height_required_per_network_output_row)
        print("closest lower number of rows: " + str(closest_lower_number_of_rows))

        second_scale_reduction_required = 1 / (closest_lower_number_of_rows / actual_number_of_rows)
        print("second scale reduction required: " + str(second_scale_reduction_required))
        result = min_scaling_factor * second_scale_reduction_required

        print("number of rows for max height:" + str((float(max_image_height) /
                                                     self.height_required_per_network_output_row) / result))

        return result, True

    def get_data_loader_with_appropriate_padding(self, data_set, max_image_height,
                                                 max_image_width, max_labels_length,
                                                 batch_size: int, padding_strategy: PaddingStrategy,
                                                 keep_unsigned_int_format: bool,
                                                 shuffle: bool,
                                                 use_four_pixel_input_blocks: bool):
        to_tensor = ToTensor()

        train_set_pairs = list([])

        # Rescale the image
        # scale_reduction_factor = 2

        # Find a scaling factor > 2 that makes the highest image fit in a multiple of
        # self.height_required_per_network_output_row
        scale_reduction_factor, found_suitable_higher_scale_reduction_factor = self.get_scale_reduction_factor_that_minimizes_horizontal_padding(max_image_height, 2)
        print("Found scale_reduction_factor >= 2 optimal for minimizing vertical padding: " +
              str(scale_reduction_factor))

        # Not only must images be padded to be all the same size, they
        # must also be of a size that is a multiple of block_size squared
        # FIXME: This is a quick and dirty hack to get the sizes right
        # Implement for general block sizes

        max_image_height = IamLinesDataset.compute_max_adjusted_by_scale_reduction_factor(max_image_height,
                                                                                          scale_reduction_factor)

        # Sanity check: after the special rescaling the max_image_height should fit
        # exactly in a multiple of self.height_required_per_network_output_row
        if found_suitable_higher_scale_reduction_factor and (max_image_height % self.height_required_per_network_output_row) > 0:
            raise RuntimeError("Error: the max_image_height " + str(max_image_height) +
                               " after rescaling should be an exact multiple of " +
                               str(self.height_required_per_network_output_row) +
                               "but it is not")

        max_image_width = IamLinesDataset.compute_max_adjusted_by_scale_reduction_factor(max_image_width,
                                                                                         scale_reduction_factor)
        max_image_height = \
            max_image_height + LastMinutePadding.get_additional_amount_required_to_make_multiple_of_value(
                max_image_height, self.height_required_per_network_output_row)

        # Width that is strictly required to fit the max occurring width and also
        # be a multiple of self.width_required_per_network_output_row

        max_image_width = max_image_width + LastMinutePadding.get_additional_amount_required_to_make_multiple_of_value(
                max_image_width, self.width_required_per_network_output_column)

        print("After scaling and addition to fit into multiple of the " +
              "height/width consumed per output row/column by the network : " +
              "\nmax_image_height: " + str(max_image_height) +
              "\nmax_image_width: " + str(max_image_width))

        sample_index = 0
        last_percentage_complete = 0

        for sample in data_set:
            print("sample index: " + str(sample_index))

            #if sample_index >= 32:   # Hack for fast testing
            #    break

            image_original = sample["image"]
            image_original_dtype = image_original.dtype

            # print("image_original: " + str(image_original))
            # print("image_original.dtype: " + str(image_original.dtype))

            image_height_original, image_width_original = image_original.shape

            # Only images with both dimensions larger than the scale reduction factor
            # are rescaled, to avoid errors in scaling
            if image_height_original > scale_reduction_factor and image_width_original > scale_reduction_factor:
                image_height = int(image_height_original / scale_reduction_factor)
                image_width = int(image_width_original / scale_reduction_factor)

                if not use_four_pixel_input_blocks:
                    rescale = Rescale(tuple([image_height, image_width]))
                    # Rescale works on ndarrays, not on pytorch tensors
                    # print("before: sample[\"image\"].dtype: " + str(sample["image"].dtype))

                    sample = rescale(sample)
            else:
                image_height = image_height_original
                image_width = image_width_original
            image_rescaled = sample["image"]
            # print("image_rescaled.dtype: " + str(image_rescaled.dtype))

            # Sanity check that the image dtype remained the same
            if image_original_dtype != image_rescaled.dtype:
                raise RuntimeError("Error: the image dtype changed")

            sample_pytorch = to_tensor(sample)

            if use_four_pixel_input_blocks:

                # Create a version of original image formed by stacking the input pixels within blocks of
                # 2 by 2 along the third (=channel) dimension
                four_pixel_block_image  = TensorBlockStacking.rescale_tensor_by_stacking_tensor_blocks(
                    sample_pytorch['image'], SizeTwoDimensional.create_size_two_dimensional(2, 2),
                    IamLinesDataset.UINT8_WHITE_VALUE)
                sample_pytorch['image'] = four_pixel_block_image
                image_height = four_pixel_block_image.size(1)
                image_width = four_pixel_block_image.size(2)



           # print("sample_pytorch: " + str(sample_pytorch))

            image = sample_pytorch["image"]
            # print(">>> image size: " + str(image.size()))

            # Get the labels and labels length
            labels = sample_pytorch["labels"]
            labels_length = labels.size(0)

            # print("image_height: " + str(image_height))
            # print("image_width: " + str(image_width))
            # print("labels_length: " + str(labels_length))

            # Use the provided function "get_collumns_padding_required_fuction" to
            # determine the columns of padding required
            columns_padding_required = padding_strategy.get_columns_padding_required(image_width, max_image_width)
            rows_padding_required = padding_strategy.get_rows_padding_required(image_height, max_image_height)
            rows_padding_required_top = int(rows_padding_required / 2)
            # Make sure no row gets lost through integer division
            rows_padding_required_bottom = rows_padding_required - rows_padding_required_top

            print("columns_padding_required: " + str(columns_padding_required))
            print("rows_padding_required: " + str(rows_padding_required))

            # See: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
            # pad last dimension (width) by 0, columns_padding_required
            # and one-but-last dimension (height) by 0, rows_padding_required
            # p2d = (0, columns_padding_required, rows_padding_required, 0)

            if use_four_pixel_input_blocks:
                p3d = (0, columns_padding_required,
                       rows_padding_required_top,
                       rows_padding_required_bottom, 0, 0)
                # We want to pad with the value for white, which is 255 for uint8
                image_padded = torch.nn.functional. \
                    pad(image, p3d, "constant", IamLinesDataset.UINT8_WHITE_VALUE)

            else:

                p2d = (0, columns_padding_required,
                       rows_padding_required_top,
                       rows_padding_required_bottom)
                # We want to pad with the value for white, which is 255 for uint8
                image_padded = torch.nn.functional. \
                    pad(image, p2d, "constant", IamLinesDataset.UINT8_WHITE_VALUE)

            # Show padded image: for debugging
            # print("image: " + str(image))

            # Show padded images with the least padding, to see if the
            # padding is really necessary
            visualization_cutoff = 0.90 * max_image_height
            #if image_height > visualization_cutoff:
            #     util.image_visualization.imshow_tensor_2d(image_padded)

            #util.image_visualization.imshow_tensor_2d(image_padded)

            # outputs_per_label = (image_width / float(self.width_required_per_network_output_column)) / labels.size(0)
            # if outputs_per_label > 15:
            #     print("outputs_per_label: " + str(outputs_per_label))
            #     print("labels: " + str(labels))
            #     util.image_visualization.imshow_tensor_2d(image)

            # Add additional bogus channel dimension, since a channel dimension is expected by downstream
            # users of this method
            print("before: image.size(): " + str(image.size()))
            if not use_four_pixel_input_blocks:
                image_padded = image_padded.unsqueeze(0)
            print("after padding: image_padded.size(): " + str(image_padded.size()))
            # print("after padding: image_padded: " + str(image_padded))

            if not keep_unsigned_int_format:
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

            percentage_complete = int((float(sample_index) / len(data_set)) * 100)

            # Print every 10% if not already printed
            if ((percentage_complete % 10) == 0) and (percentage_complete != last_percentage_complete):
                print("iam_lines_dataset.get_data_loader_with_appropriate_padding - completed " +
                      str(percentage_complete) + "%")
                sys.stdout.flush()
                # print(" sample index: " + str(sample_index) + " len(data_set):" +
                #      str(len(data_set)))
                last_percentage_complete = percentage_complete
            sample_index += 1

        data_loader = padding_strategy.create_data_loader(train_set_pairs, batch_size,
                                                           shuffle)
        return data_loader

    @staticmethod
    def check_fractions_add_up_to_one(fractions: list):
        total_fractions = 0
        for fraction in fractions:
            total_fractions += fraction

        # Check that the specified fractions add up to one
        if not total_fractions == 1:
            raise RuntimeError("Error: fractions" + str(fractions) +
                               " must sum up to one")

    def get_train_set_validation_set_test_set_data_loaders(
            self, batch_size: int, train_set, validation_set, test_set,
            minimize_vertical_padding: bool,
            minimize_horizontal_padding: bool, keep_unsigned_int_format: bool,
            perform_horizontal_batch_padding_in_data_loader_: bool,
            use_four_pixel_input_blocks: bool):

        print("Entered get_random_train_set_validation_set_test_set_data_loaders...")

        max_image_height, max_image_width = self.get_max_image_dimension()
        # print("max image height: " + str(max_image_height))
        # print("max image width: " + str(max_image_width))
        max_labels_length = self.get_max_labels_length()
        # print("max labels length: " + str(max_labels_length))

        padding_strategy = PaddingStrategy.create_padding_strategy(self.height_required_per_network_output_row,
                                                                   self.width_required_per_network_output_column,
                                                                   minimize_vertical_padding,
                                                                   minimize_horizontal_padding,
                                                                   perform_horizontal_batch_padding_in_data_loader_)

        print("Prepare IAM data train loader...")
        train_loader = self.get_data_loader_with_appropriate_padding(
            train_set, max_image_height, max_image_width, max_labels_length, batch_size, padding_strategy,
            keep_unsigned_int_format, shuffle=True, use_four_pixel_input_blocks=use_four_pixel_input_blocks)

        print("Prepare IAM data validation loader...")
        validation_loader = self.get_data_loader_with_appropriate_padding(
            validation_set, max_image_height, max_image_width, max_labels_length, batch_size,
            padding_strategy, keep_unsigned_int_format, shuffle=False,
            use_four_pixel_input_blocks=use_four_pixel_input_blocks)

        print("Prepare IAM data test loader...")
        test_loader = self.get_data_loader_with_appropriate_padding(
            test_set, max_image_height, max_image_width, max_labels_length, batch_size, padding_strategy,
            keep_unsigned_int_format, shuffle=False, use_four_pixel_input_blocks=use_four_pixel_input_blocks)

        return train_loader, validation_loader, test_loader

    @staticmethod
    def write_iam_dataset_to_file(
            output_file_path: str, iam_lines_dataset):

        with open(output_file_path, "w") as output_file:
            for iam_line_information in iam_lines_dataset.examples_line_information:
                output_file.write(
                    "".join(iam_line_information.get_characters_with_word_separator()) + "\n")

    def get_random_train_set_validation_set_test_set_data_loaders(
            self, batch_size: int, train_examples_fraction: float,
            validation_examples_fraction: float, test_examples_fraction: float,
            permutation_save_or_load_file_path: str,
            dataset_save_or_load_file_path: str,
            minimize_vertical_padding: bool,
            minimize_horizontal_padding: bool, keep_unsigned_int_format: bool,
            perform_horizontal_batch_padding_in_data_loader: bool,
            save_dev_set_file_path: str, save_test_set_file_path: str
    ):

        if os.path.isfile(dataset_save_or_load_file_path):
            return self.load_dataset_from_file(dataset_save_or_load_file_path)

        IamLinesDataset.check_fractions_add_up_to_one(list([train_examples_fraction,
                                                            validation_examples_fraction,
                                                            test_examples_fraction]))

        train_set, validation_set, test_set = self. \
            split_random_train_set_validation_set_and_test_set(train_examples_fraction,
                                                               validation_examples_fraction,
                                                               permutation_save_or_load_file_path)

        if save_dev_set_file_path is not None:
            IamLinesDataset.write_iam_dataset_to_file(save_dev_set_file_path,
                                                      validation_set)

        if save_test_set_file_path is not None:
            IamLinesDataset.write_iam_dataset_to_file(save_test_set_file_path,
                                                      test_set)

        train_loader, validation_loader, test_loader = self.get_train_set_validation_set_test_set_data_loaders(
            batch_size, train_set, validation_set, test_set,
            minimize_vertical_padding,
            minimize_horizontal_padding, keep_unsigned_int_format,
            perform_horizontal_batch_padding_in_data_loader)

        self.save_dataset_to_file(dataset_save_or_load_file_path, train_loader, validation_loader, test_loader)

        return train_loader, validation_loader, test_loader

    @staticmethod
    def get_iam_example_ids_set_from_split_file(split_file_path: str):
        result = set([])
        with open(split_file_path, "r") as file:
            for line in file:
                example_id = line.strip()
                result.add(example_id)
        print("get_iam_example_ids_set_from_split_file - split_file_path: " +
              split_file_path + " result:\n" + str(result))
        return result

    @staticmethod
    def get_set_containing_element_and_check_it_is_unique(sets: list, element):
        """
        Method that finds the set that contains the element from a list of sets,
        prints a warning if the set is not found,and raises an error if  is not unique.

        :param sets:
        :param element:
        :return:
        """
        matching_set = None
        for i in range(0, len(sets)):
            current_set = sets[i]
            if element in current_set:
                if matching_set is not None:
                    raise RuntimeError("Error: found multiple matching sets for :" + str(element))
                matching_set = current_set
        if matching_set is None:
            print("WARNING: found no matching set for: " + str(element))
        return matching_set

    @staticmethod
    def get_relevant_part_line_id(iam_line_information):
        line_id_parts = iam_line_information.line_id.split("-")
        relevant_part_line_id = line_id_parts[0] + "-" + line_id_parts[1]
        return relevant_part_line_id

    def split_specified_train_set_validation_set_and_test_set(
            self, train_example_ids_set: set, dev_example_ids_set: set,
            test_example_ids_set: set):

        print("Entered split_specified_train_set_validation_set_and_test_set...")

        # print("last_train_index: " + str(last_train_index))
        examples_line_information_train = list([])
        examples_line_information_validation = list([])
        examples_line_information_test = list([])
        examples_line_information_omitted = list([])

        sets_list = list([train_example_ids_set, dev_example_ids_set, test_example_ids_set])
        for iam_line_information in self.examples_line_information:
            relevant_part_line_id = \
                IamLinesDataset.get_relevant_part_line_id(iam_line_information)
            set_containing_id = IamLinesDataset.get_set_containing_element_and_check_it_is_unique(
                sets_list, relevant_part_line_id)
            if set_containing_id is train_example_ids_set:
                print("Assigning example with id" + str(iam_line_information.line_id) + " to training set")
                examples_line_information_train.append(iam_line_information)
            elif set_containing_id is dev_example_ids_set:
                print("Assigning example with id" + str(iam_line_information.line_id) + " to validation set")
                examples_line_information_validation.append(iam_line_information)
            elif set_containing_id is test_example_ids_set:
                print("Assigning example with id" + str(iam_line_information.line_id) + " to test set")
                examples_line_information_test.append(iam_line_information)
            else:
                print("Warning: did not find a valid ids set for the example " + iam_line_information.line_id +
                      " - omitting this example ")
                examples_line_information_omitted.append(iam_line_information)
        print(">>> number of training examples: " + str(len(examples_line_information_train)))
        print(">>> number of development examples: " + str(len(examples_line_information_validation)))
        print(">>> number of test examples: " + str(len(examples_line_information_test)))
        print(">>> number of omitted examples: " + str(len(examples_line_information_omitted)))

        train_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_train,
                                    self.string_to_index_mapping_table, 64, 8, self.transform)
        validation_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_validation,
                                         self.string_to_index_mapping_table, 64, 8, self.transform)
        test_set = IamLinesDataset(self.iam_lines_dictionary, examples_line_information_test,
                                   self.string_to_index_mapping_table, 64, 8, self.transform)

        return train_set, validation_set, test_set

    def load_dataset_from_file(self, dataset_load_file_path: str):
            print(">>>Loading data from saved file \"" + dataset_load_file_path + "\"...")
            train_loader, validation_loader, test_loader = torch.load(dataset_load_file_path)
            print("done.")
            return train_loader, validation_loader, test_loader

    def save_dataset_to_file(self, dataset_save_file_path, train_loader, validation_loader, test_loader):
        print(">>>Saving data to \"" + dataset_save_file_path + "\"...")
        torch.save((train_loader, validation_loader, test_loader), dataset_save_file_path)
        print("done.")

    def get_train_set_validation_set_test_set_data_loaders_using_split_specification_files(
            self, batch_size: int, train_examples_split_file_path: str,
            dev_examples_split_file_path: str, test_examples_split_file_path: str,
            minimize_vertical_padding: bool,
            minimize_horizontal_padding: bool, keep_unsigned_int_format: bool,
            perform_horizontal_batch_padding_in_data_loader: bool,
            use_four_pixel_input_blocks: bool,
            dataset_save_or_load_file_path: str
    ):

        if os.path.isfile(dataset_save_or_load_file_path):
            return self.load_dataset_from_file(dataset_save_or_load_file_path)

        train_example_ids_set = IamLinesDataset.get_iam_example_ids_set_from_split_file(train_examples_split_file_path)
        dev_example_ids_set = IamLinesDataset.get_iam_example_ids_set_from_split_file(dev_examples_split_file_path)
        test_example_ids_set = IamLinesDataset.get_iam_example_ids_set_from_split_file(test_examples_split_file_path)

        train_set, validation_set, test_set = self.split_specified_train_set_validation_set_and_test_set(
            train_example_ids_set, dev_example_ids_set, test_example_ids_set)

        train_loader, validation_loader, test_loader =  self.get_train_set_validation_set_test_set_data_loaders(
            batch_size, train_set, validation_set, test_set,
            minimize_vertical_padding,
            minimize_horizontal_padding, keep_unsigned_int_format,
            perform_horizontal_batch_padding_in_data_loader,
            use_four_pixel_input_blocks=use_four_pixel_input_blocks)

        self.save_dataset_to_file(dataset_save_or_load_file_path, train_loader, validation_loader, test_loader)

        return train_loader, validation_loader, test_loader

    def __len__(self):
        return len(self.examples_line_information)
        # return int(len(self.examples_line_information) / 300)  # Hack for faster training during development
        # return 32  # Hack for faster training during development

    def __getitem__(self, idx):
        line_information = self.examples_line_information[idx]
        # print("__getitem__ line_information: " + str(line_information))
        # image_file_path = self.iam_lines_dictionary.get_image_file_path(line_information)
        # print("image_file_path: " + str(image_file_path))

        sample = {'image': self.get_image(idx), 'labels': self.get_labels(idx)}

        # rescale = Rescale(tuple([10, 10]))
        # rescale(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, index):
        line_information = self.examples_line_information[index]
        print("__getitem__ line_information: " + str(line_information))
        image_file_path = self.iam_lines_dictionary.get_image_file_path(line_information)
        # print("image_file_path: " + str(image_file_path))

        image = io.imread(image_file_path)
        # print(">>> get_image - image.dtype: " + str(image.dtype))
        return image

    def get_labels(self, index):
        line_information = self.examples_line_information[index]
        # print("__getitem__ line_information: " + str(line_information))
        image_file_path = self.iam_lines_dictionary.get_image_file_path(line_information)
        # print("image_file_path: " + str(image_file_path))

        characters = line_information.get_characters_with_word_separator()
        indices = self.string_to_index_mapping_table.get_indices(characters)

        indices_array = numpy.ndarray((len(indices)), buffer=numpy.array(indices), dtype=int)

        return indices_array

    # Print statistics about the size of the images in the dataset. This useful to get an idea about how much
    # computation we "waste" by padding everything to the maximum width and maximum height
    @staticmethod
    def print_image_dimension_statistics(min_height, max_height, mean_height, min_width, max_width, mean_width,
                                         min_outputs_per_label, max_outputs_per_label, mean_outputs_per_label):
        print(">>> IamLinesDataset.get_max_image_dimensions: ")
        print(" min_height: " + str(min_height) + " max_height: " + str(max_height) +
              " mean_height: " + str(mean_height))
        print(" min_width: " + str(min_width) + " max_width: " + str(max_width) +
              " mean_width: " + str(mean_width))
        print("Outputs per label statistics before any input downscaling:")
        print(" min_outputs_per_label: " + str(min_outputs_per_label) + " max_outputs_per_label: "
              + str(max_outputs_per_label) +
              " mean_outputs_per_label: " + str(mean_outputs_per_label))

    def get_max_image_dimension(self):
        # Initialize the minimum to the maximum integer value
        min_height = sys.maxsize
        max_height = 0
        summed_heights = 0

        # Initialize the minimum to the maximum integer value
        min_width = sys.maxsize
        max_width = 0
        summed_widths = 0

        # Statistics for number of outputs
        min_outputs_per_label = sys.maxsize
        max_outputs_per_label = 0
        summed_outputs_per_label = 0

        for index in range(0, self.__len__()):
            image = self.get_image(index)
            height, width = image.shape

            number_of_labels = len(self.get_labels(index))
            outputs = width / float(self.width_required_per_network_output_column)
            outputs_per_label = outputs / number_of_labels

            min_height = min(min_height, height)
            min_width = min(min_width, width)
            min_outputs_per_label = min(min_outputs_per_label, outputs_per_label)

            max_height = max(max_height, height)
            max_width = max(max_width, width)
            max_outputs_per_label = max(max_outputs_per_label, outputs_per_label)

            summed_heights += height
            summed_widths += width
            summed_outputs_per_label += outputs_per_label

        mean_height = summed_heights / float(self.__len__())
        mean_width = summed_widths / float(self.__len__())
        mean_outputs_per_label = summed_outputs_per_label / float(self.__len__())

        IamLinesDataset.print_image_dimension_statistics(min_height, max_height, mean_height,
                                                         min_width, max_width, mean_width,
                                                         min_outputs_per_label, max_outputs_per_label,
                                                         mean_outputs_per_label)

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

    def split_random_train_set_validation_set_and_test_set_default_subset_size_fractions(
            self, permutation_save_or_load_file_path: str):
        return self.split_random_train_set_validation_set_and_test_set(
            IamLinesDataset.TRAIN_EXAMPLES_FRACTION, IamLinesDataset.VALIDATION_EXAMPLES_FRACTION,
            permutation_save_or_load_file_path)


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

        image_type_original = image.dtype

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # img = transform.resize(image, (new_h, new_w))

        # print("rescale - image.dtype: " + str(image.dtype))

        # The option "preserve_range=True is crucial for preserving
        # ints, and not converting to floats
        # See: http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
        image_result = transform.resize(image, (new_h, new_w),
                               mode="constant", anti_aliasing=True,
                               preserve_range=True)

        # print("rescale - img_result.dtype: " + str(image_result.dtype))

        # Still need to convert back to the original type, since
        # transform has no option to preserve type. "unsafe" option is
        # needed to allow casting floats to ints
        # See: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.astype.html
        image_result_converted_back = image_result.\
            astype(image_type_original, casting="unsafe")

        return {'image': image_result_converted_back, 'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))

        # print("ToTensor.call - image.dtype: " + str(image.dtype))
        # Appearantly the right torch tensor type is determined automatically
        # https://github.com/pytorch/pytorch/issues/541
        torch_image = torch.from_numpy(image)
        # print("torch_image.size(): " + str(torch_image.size()))
        # FIXME: labels must be an np.ndarray
        return {'image': torch_image,
                'labels': torch.from_numpy(labels).type(torch.IntTensor)}


def test_iam_lines_dataset():
    lines_file_path = "/datastore/data/iam-database/ascii/lines.txt"
    iam_database_line_images_root_folder_path = "/datastore/data/iam-database/lines"
    iam_lines_dicionary = IamExamplesDictionary.create_iam_lines_dictionary(lines_file_path,
                                                                            iam_database_line_images_root_folder_path,
                                                                            True)
    iam_lines_dataset = IamLinesDataset.create_iam_dataset(iam_lines_dicionary,
                                                           "./string_to_index_mapping_table_test.txt",
                                                           "ok",
                                                           None)
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

    permutation_save_or_load_file_path = "test_permutation_file.txt"
    minimize_horizontal_padding = True
    data_loader_perform_horizontal_batch_padding = False
    iam_lines_dataset.\
        get_random_train_set_validation_set_test_set_data_loaders(16, 0.5, 0.2, 0.3,
                                                                  permutation_save_or_load_file_path,
                                                                  minimize_horizontal_padding, False,
                                                                  data_loader_perform_horizontal_batch_padding
                                                                  )


def test_iam_words_dataset():
    words_file_path = "/datastore/data/iam-database/ascii/words.txt"
    iam_database_word_images_root_folder_path = "/datastore/data/iam-database/words"
    iam_words_dicionary = IamExamplesDictionary.create_iam_words_dictionary(words_file_path,
                                                                            iam_database_word_images_root_folder_path,
                                                                            False)
    iam_lines_dataset = IamLinesDataset.create_iam_dataset(iam_words_dicionary,
                                                           "./string_to_index_mapping_table_test.txt", "ok", None)
    sample = iam_lines_dataset[0]
    print("iam_words_dataset[0]: " + str(sample))
    print("(iam_words_dataset[0])[image]: " + str(sample["image"]))
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

    permutation_save_or_load_file_path = "test_permutation_file3.txt"
    minimize_horizontal_padding = True
    data_loader_perform_horizontal_batch_padding = False
    iam_lines_dataset.\
        get_random_train_set_validation_set_test_set_data_loaders(16, 0.5, 0.2, 0.3,
                                                                  permutation_save_or_load_file_path,
                                                                  minimize_horizontal_padding, False,
                                                                  data_loader_perform_horizontal_batch_padding
                                                                  )


def main():
    # test_iam_lines_dataset()
    test_iam_words_dataset()


if __name__ == "__main__":
    main()
