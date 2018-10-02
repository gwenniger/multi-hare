import sys
import re
from data_preprocessing.iam_database_preprocessing.iam_examples_dictionary import IamLineInformation
from data_preprocessing.iam_database_preprocessing.iam_examples_dictionary import IamExamplesDictionary
from data_preprocessing.iam_database_preprocessing.iam_dataset import IamLinesDataset
from util.utils import Utils
from collections import OrderedDict


class IamLinesToCorpusLinesMapper:

    def __init__(self, iam_lines_without_spaces_line_id_tuple_list: list,
                 lob_corpus_lines_without_whitespace_list: list,
                 lob_corpus_lines_list: list):
        self.iam_lines_without_spaces_line_id_tuple_list = iam_lines_without_spaces_line_id_tuple_list
        self.lob_corpus_lines_without_whitespace_list = lob_corpus_lines_without_whitespace_list
        self.lob_corpus_lines_list = lob_corpus_lines_list
        self.iam_line_id_to_corpus_index_mapping_table = dict([])

    @staticmethod
    def create_line_without_whitespace(line):
        line_without_whitespace = re.sub("\s+", "", line)
        return line_without_whitespace

    @staticmethod
    def create_iam_lines_to_corpus_lines_mapper(
            iam_database_lines_file_path: str, iam_database_line_images_root_folder_path: str,
            permutation_save_or_load_file_path: str, vocabulary_file_path: str,
            lob_corpus_file_path: str):
        validation_and_test_set_sorted = \
            IamLinesToCorpusLinesMapper.\
            create_iam_lines_without_spaces_line_id_tuple_list(
                iam_database_lines_file_path, iam_database_line_images_root_folder_path,
                vocabulary_file_path)
        lob_corpus_lines_list = list([])
        lob_corpus_lines_without_whitespace_list = list([])
        with open(lob_corpus_file_path, 'r') as file:
            for line in file:
                line_without_whitespace = IamLinesToCorpusLinesMapper.create_line_without_whitespace(line)
                lob_corpus_lines_list.append(line)
                lob_corpus_lines_without_whitespace_list.append(line_without_whitespace)

        return IamLinesToCorpusLinesMapper(validation_and_test_set_sorted,
                                           lob_corpus_lines_without_whitespace_list,
                                           lob_corpus_lines_list)

    @staticmethod
    def create_iam_lines_without_spaces_line_id_tuple_list(
            iam_database_lines_file_path: str, iam_database_line_images_root_folder_path: str,
            vocabulary_file_path: str
    ):
        result = list([])

        iam_lines_dataset = IamLinesDataset.create_iam_lines_dataset_from_input_files(
            iam_database_lines_file_path, iam_database_line_images_root_folder_path, vocabulary_file_path
        )

        for iam_line_information in iam_lines_dataset.examples_line_information:
            print("iam_line_information.line_id: " + str(iam_line_information.line_id))
            characters_as_string = "".join(iam_line_information.get_characters())
            characters_as_string_no_whitespace =\
                IamLinesToCorpusLinesMapper.create_line_without_whitespace(characters_as_string)
            print("line characters_as_string_no_whitespace: "
                  + str(characters_as_string_no_whitespace))
            result.append(tuple([characters_as_string_no_whitespace, iam_line_information.line_id]))
        return result

    def find_first_matching_corpus_line_index_searching_from_index(self,
                                                                   iam_line_index, index_offset):
        iam_line_line_id_tuple = self.iam_lines_without_spaces_line_id_tuple_list[iam_line_index]
        line_without_whitespace = iam_line_line_id_tuple[0]

        index = index_offset
        for corpus_line_without_whitespace in self.lob_corpus_lines_without_whitespace_list[index_offset:100]:
            print("corpus_line_without_whitespace: " + corpus_line_without_whitespace)
            if line_without_whitespace in corpus_line_without_whitespace:
                return index
            index += 1
        raise RuntimeError("Error: line " + line_without_whitespace + " not found in remaining corpus lines")

    def generate_iam_line_id_to_corpus_index_mapping_table(self):

        result = dict([])

        last_matched_index = -1
        current_corpus_line_index = 0

        for i in range(0, len(self.iam_lines_without_spaces_line_id_tuple_list)):

            first_matching_index = \
                self.find_first_matching_corpus_line_index_searching_from_index(i, current_corpus_line_index)
            if not first_matching_index == last_matched_index or (first_matching_index == (last_matched_index + 1)):
                first_matching_index_next = \
                    self.find_first_matching_corpus_line_index_searching_from_index(i + 1, first_matching_index)
                if not first_matching_index_next == first_matching_index or \
                        first_matching_index_next == (first_matching_index + 1):
                    raise RuntimeError("Did not manage to find consecutive matches")

            iam_line_line_id_tuple = self.iam_lines_without_spaces_line_id_tuple_list[i]
            # line_without_whitespace = iam_line_line_id_tuple[0]
            id = iam_line_line_id_tuple[1]

            result[id] = first_matching_index
        return result


class IamDatabaseFragmentsRemover:
    """
    This class takes care of removing text fragments that occur in the
    IAM Database training, validation or testing material from the the
    data used to train a language model.

    """

    def __init__(self, iam_lines_to_corpus_lines_mapper: IamLinesToCorpusLinesMapper,
                 corpus_input_file_path: str,
                 corpus_output_file_path: str
                 ):
        self.iam_lines_to_corpus_lines_mapper = iam_lines_to_corpus_lines_mapper
        self.corpus_input_file_path = corpus_input_file_path
        self.corpus_output_file_path = corpus_output_file_path

    @staticmethod
    def create_iam_database_fragments_remover(
            iam_lines_file_path: str,
            iam_database_line_images_root_folder_path: str,
            corpus_input_file_path: str,
            corpus_output_file_path: str,
            permutation_file_path: str,
            vocabulary_file_path: str):
        iam_lines_to_corpus_lines_mapper = IamLinesToCorpusLinesMapper.create_iam_lines_to_corpus_lines_mapper(
            iam_lines_file_path, iam_database_line_images_root_folder_path, permutation_file_path, vocabulary_file_path,
            corpus_input_file_path)
        print("iam_lines_without_spaces_line_id_tuple_list: " +
              str(iam_lines_to_corpus_lines_mapper.iam_lines_without_spaces_line_id_tuple_list))
        iam_lines_to_corpus_lines_mapper.generate_iam_line_id_to_corpus_index_mapping_table()

        return IamDatabaseFragmentsRemover(
            iam_lines_to_corpus_lines_mapper, corpus_input_file_path, corpus_output_file_path)



    def create_iam_lines_filtered_output_file(self):

        with open(self.corpus_input_file_path, 'r') as file:
            with open(self.corpus_output_file_path, 'w') as output_file:
                for line in file:
                    line_without_whitespace = re.sub("\s+", "", line)

                    if line_without_whitespace in self.iam_lines_without_spaces:
                        print("String: \"" + line_without_whitespace + "\" is part of iam data")
                    #else:
                        # print("String " + line_without_whitespace + " is not part of iam data")


def main():

    if len(sys.argv) != 7:
        print("number of arguments: " + str(len(sys.argv)))
        raise RuntimeError("Error - usage: "
                           "iam_database_fragments_remover IAM_LINES_FILE_PHAT " 
                           "IAM_DATABASE_LINE_IMAGES_ROOT_FOLDER_PATH "
                           "CORPUS_INPUT_FILE_PATH "
                           "CORPUS_OUTPUT_FILE_PATH "
                           "PERMUTATION_FILE_PATH"
                           "VOCABULARY_FILE_PATH")

    iam_lines_file_path = sys.argv[1]
    print("iam_lines_file_path: " + iam_lines_file_path)
    iam_database_line_images_root_folder_path = sys.argv[2]
    print("iam_database_line_images_root_folder_path: " + iam_database_line_images_root_folder_path)
    corpus_input_file_path = sys.argv[3]
    corpus_output_file_path = sys.argv[4]
    permutation_file_path = sys.argv[5]
    vocabulary_file_path = sys.argv[6]
    iam_database_fragments_remover = IamDatabaseFragmentsRemover.create_iam_database_fragments_remover(
        iam_lines_file_path, iam_database_line_images_root_folder_path,
        corpus_input_file_path, corpus_output_file_path,
        permutation_file_path,
        vocabulary_file_path
    )
    # iam_database_fragments_remover.create_iam_lines_filtered_output_file()


if __name__ == "__main__":
    main()