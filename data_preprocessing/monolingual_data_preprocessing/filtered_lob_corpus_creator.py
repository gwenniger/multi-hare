import sys
import re
from data_preprocessing.iam_database_preprocessing.iam_dataset import IamLinesDataset
from data_preprocessing.monolingual_data_preprocessing.lob_original_preprocessor import LobOriginalPreprocessor


class FilteredLobCorpusCreator:

    def __init__(self, iam_validation_and_test_set_fragment_ids: set,
                 lob_original_preprocessor: LobOriginalPreprocessor,
                 corpus_output_file_path: str):
        self.iam_validation_and_test_set_fragment_ids = iam_validation_and_test_set_fragment_ids
        self.lob_original_preprocessor = lob_original_preprocessor
        self.corpus_output_file_path = corpus_output_file_path

    @staticmethod
    def create_line_without_whitespace(line):
        line_without_whitespace = re.sub("\s+", "", line)
        return line_without_whitespace

    @staticmethod
    def create_filterd_lob_corpus_creator(
            iam_database_lines_file_path: str, iam_database_line_images_root_folder_path: str,
            permutation_save_or_load_file_path: str, vocabulary_file_path: str,
            lob_input_files_directory: str, corpus_output_file_path: str):
        validation_and_test_set_iam_ids = \
            FilteredLobCorpusCreator.create_iam_validation_and_test_set_iam_fragment_ids_set(
                iam_database_lines_file_path, iam_database_line_images_root_folder_path,
                vocabulary_file_path, permutation_save_or_load_file_path)
        lob_original_preprocessor = LobOriginalPreprocessor.create_lob_original_preprocessor(lob_input_files_directory)
        return FilteredLobCorpusCreator(validation_and_test_set_iam_ids, lob_original_preprocessor,
                                        corpus_output_file_path)

    @staticmethod
    def create_iam_validation_and_test_set_iam_fragment_ids_set(
            iam_database_lines_file_path: str, iam_database_line_images_root_folder_path: str,
            vocabulary_file_path: str, permutation_save_or_load_file_path: str
    ):

        iam_lines_dataset = IamLinesDataset.create_iam_lines_dataset_from_input_files(
            iam_database_lines_file_path, iam_database_line_images_root_folder_path, vocabulary_file_path
        )
        train_set, validation_set, test_set = \
            iam_lines_dataset.split_random_train_set_validation_set_and_test_set_default_subset_size_fractions(
                permutation_save_or_load_file_path)
        validation_and_test_set_iam_ids = set([])
        validation_and_test_set = list([])
        validation_and_test_set.extend(validation_set.examples_line_information)
        validation_and_test_set.extend(test_set.examples_line_information)
        for item in validation_and_test_set:
            line_id = item.line_id
            iam_framgment_id = line_id.split("-")[0]
            validation_and_test_set_iam_ids.add(iam_framgment_id)
        return validation_and_test_set_iam_ids

    def create_iam_validation_and_test_fragments_filtered_output_file(self):

        with open(self.corpus_output_file_path, 'w') as output_file:
            lob_part_id_to_lines_map =  self.lob_original_preprocessor.process_lob_original_files()

            for key in lob_part_id_to_lines_map:
                if key not in self.iam_validation_and_test_set_fragment_ids:
                    print("Including lines for iam fragment with key : " + key)
                    # output_file.write("<key: " + str(key) + ">" + "\n")
                    for line in lob_part_id_to_lines_map[key]:
                        output_file.write(line + "\n")
                else:
                    print("Omitting iam fragment with key : " + key +
                          " because it appeared in the validation or test set")


def main():

    if len(sys.argv) != 7:
        print("number of arguments: " + str(len(sys.argv)))
        raise RuntimeError("Error - usage: "
                           "iam_database_fragments_remover IAM_LINES_FILE_PHAT " 
                           "IAM_DATABASE_LINE_IMAGES_ROOT_FOLDER_PATH "
                           "IAM_ORIGINAL_FILES_DIRECTORY_PATH "
                           "CORPUS_OUTPUT_FILE_PATH "
                           "PERMUTATION_FILE_PATH"
                           "VOCABULARY_FILE_PATH")

    iam_lines_file_path = sys.argv[1]
    print("iam_lines_file_path: " + iam_lines_file_path)
    iam_database_line_images_root_folder_path = sys.argv[2]
    print("iam_database_line_images_root_folder_path: " + iam_database_line_images_root_folder_path)
    iam_original_files_directory_path = sys.argv[3]
    corpus_output_file_path = sys.argv[4]
    permutation_file_path = sys.argv[5]
    vocabulary_file_path = sys.argv[6]
    filtered_lob_corpus_creator = FilteredLobCorpusCreator.create_filterd_lob_corpus_creator(
        iam_lines_file_path, iam_database_line_images_root_folder_path,
        permutation_file_path,
        vocabulary_file_path,
        iam_original_files_directory_path,
        corpus_output_file_path
    )
    filtered_lob_corpus_creator.create_iam_validation_and_test_fragments_filtered_output_file()


if __name__ == "__main__":
    main()