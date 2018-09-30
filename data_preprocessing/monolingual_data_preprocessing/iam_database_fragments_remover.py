import sys
import re
from data_preprocessing.iam_database_preprocessing.iam_examples_dictionary import IamLineInformation
from data_preprocessing.iam_database_preprocessing.iam_examples_dictionary import IamExamplesDictionary
from util.utils import Utils
from collections import OrderedDict


class IamDatabaseFragmentsRemover:
    """
    This class takes care of removing text fragments that occur in the
    IAM Database training, validation or testing material from the the
    data used to train a language model.

    """

    def __init__(self, iam_lines_without_spaces: set,
                 corpus_input_file_path: str,
                 corpus_output_file_path: str
                 ):
        self.iam_lines_without_spaces = iam_lines_without_spaces
        self.corpus_input_file_path = corpus_input_file_path
        self.corpus_output_file_path = corpus_output_file_path

    @staticmethod
    def create_iam_database_fragments_remover(
            iam_lines_file_path: str,
            iam_database_line_images_root_folder_path: str,
            corpus_input_file_path: str,
            corpus_output_file_path: str,
            use_only_ok_iam_examples_str: str):
        use_only_ok_iam_examples = Utils.str2bool(use_only_ok_iam_examples_str)
        iam_lines_dictionary = IamExamplesDictionary.create_iam_lines_dictionary(
            iam_lines_file_path, iam_database_line_images_root_folder_path, True)
        iam_lines_without_spaces = IamDatabaseFragmentsRemover.\
            create_iam_lines_without_spaces_set(iam_lines_dictionary,
                                                use_only_ok_iam_examples)
        return IamDatabaseFragmentsRemover(
            iam_lines_without_spaces, corpus_input_file_path, corpus_output_file_path)


    @staticmethod
    def get_iam_line_items(iam_lines_dictionary: IamExamplesDictionary,
                                            use_only_ok_iam_examples: bool):
        if use_only_ok_iam_examples:
            return iam_lines_dictionary.get_ok_examples()
        else:
            return iam_lines_dictionary.get_all_examples()

    @staticmethod
    def create_iam_lines_without_spaces_set(iam_lines_dictionary: IamExamplesDictionary,
                                            use_only_ok_iam_examples: bool):
        result = ""

        iam_line_items = IamDatabaseFragmentsRemover.\
            get_iam_line_items(iam_lines_dictionary, use_only_ok_iam_examples)

        for item in iam_line_items:
            iam_line_information = item[1]
            characters_as_string = "".join(iam_line_information.get_characters())
            print("line characters_as_string: " + str(characters_as_string))
            result += characters_as_string
        return result

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

    if len(sys.argv) != 6:
        print("number of arguments: " + str(len(sys.argv)))
        raise RuntimeError("Error - usage: "
                           "iam_database_fragments_remover IAM_LINES_FILE_PHAT " 
                           "IAM_DATABASE_LINE_IMAGES_ROOT_FOLDER_PATH "
                           "CORPUS_INPUT_FILE_PATH "
                           "CORPUS_OUTPUT_FILE_PATH "
                           "USE_ONLY_OK_IAM_EXAMPLES")

    iam_lines_file_path = sys.argv[1]
    iam_database_line_images_root_folder_path = sys.argv[2]
    corpus_input_file_path = sys.argv[3]
    corpus_output_file_path = sys.argv[4]
    use_only_ok_iam_examples_str = sys.argv[5]
    iam_database_fragments_remover = IamDatabaseFragmentsRemover.create_iam_database_fragments_remover(
        iam_lines_file_path, iam_database_line_images_root_folder_path,
        corpus_input_file_path, corpus_output_file_path,
        use_only_ok_iam_examples_str
    )
    iam_database_fragments_remover.create_iam_lines_filtered_output_file()


if __name__ == "__main__":
    main()