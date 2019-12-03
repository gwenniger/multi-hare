import sys
from data_preprocessing.monolingual_data_preprocessing.word_frequency_table import WordFrequencyTable
from util.utils import Utils

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class VocabularyWordCoverageAnalysis:
    """
    The purpose of this class is to study how the word coverage of the validation-set
    increases with a bigger vocabulary size, to answer the question what vocabulary
    size is necessary to get good coverage of the test-set.
    Additionally the class helps to collect and display the set of words in the
    test-set that is unknown in the entire training-set, even with no limit on
    the vocabulary size.

    """

    def __init__(self, language_model_word_frequency_table: WordFrequencyTable,
                 test_set_word_frequency_table: WordFrequencyTable):
        self.language_model_word_frequency_table = language_model_word_frequency_table
        self.test_set_word_frequency_table = test_set_word_frequency_table

    @staticmethod
    def create_vocabulary_word_coverage_analysis(language_model_training_file_path: str,
                                                 test_set_file_path: str,
                                                 collapse_word_casing: bool):
        language_model_word_frequency_table = WordFrequencyTable.create_word_frequency_table(
            language_model_training_file_path, collapse_word_casing)
        test_set_word_frequency_table = WordFrequencyTable.create_word_frequency_table(
            test_set_file_path, collapse_word_casing, "|")
        return VocabularyWordCoverageAnalysis(
            language_model_word_frequency_table, test_set_word_frequency_table)

    @staticmethod
    def get_type_token_coverage_string(
            vocabulary_size: int,
            types_total_test_set: int, tokens_total_test_set,
            types_covered: int, tokens_covered: int):
        type_coverage_percentage = (float(types_covered) / types_total_test_set) * 100
        token_coverage_percentage = (float(tokens_covered) / tokens_total_test_set) * 100
        result = str(vocabulary_size) + "," + str(types_covered) + "," + str(type_coverage_percentage) + \
            "," + str(tokens_covered) + "," + str(token_coverage_percentage)
        return result

    def make_coverage_for_vocabulary_sizes_table(self):
        types_total_test_set = len(self.test_set_word_frequency_table.word_group_frequency_table.keys())
        tokens_total_test_set = self.test_set_word_frequency_table.get_total_word_count()

        type_coverage_table = list([])
        token_coverage_table = list([])

        cumulative_type_coverage = 0
        cumulative_token_coverage = 0
        last_coverage_increase_index = -1

        uncovered_words = set([])
        for key in self.test_set_word_frequency_table.word_group_frequency_table.keys():
            uncovered_words.add(key)

        for i, word_group_frequency_tuple in enumerate(self.language_model_word_frequency_table.word_groups_sorted_by_count):
            word_group = word_group_frequency_tuple[0]
            # print("word_group: " + str(word_group))
            if word_group in self.test_set_word_frequency_table.word_group_frequency_table:
                test_set_word_frequency = self.test_set_word_frequency_table.word_group_frequency_table[
                    word_group]
                # print("test_set_word_frequency: " + str(test_set_word_frequency))
                cumulative_type_coverage += 1
                cumulative_token_coverage += test_set_word_frequency
                last_coverage_increase_index = i
                uncovered_words.remove(word_group)
            type_coverage_table.append(cumulative_type_coverage)
            token_coverage_table.append(cumulative_token_coverage)

        for i, (types_covered, tokens_covered) in enumerate(zip(type_coverage_table, token_coverage_table)):
            # print("types covered: " + str(types_covered))
            # print("tokens covered: " + str(tokens_covered))
            # print("tokens_total_test_set: " + str(tokens_total_test_set))
            print(VocabularyWordCoverageAnalysis.get_type_token_coverage_string(
                i, types_total_test_set, tokens_total_test_set, types_covered, tokens_covered) + "\n")

        # print("self.test_set_word_frequency_table.word_group_frequency_table: \n" +
        #       str(self.test_set_word_frequency_table.word_group_frequency_table))
        print("Maximum coverage of the test set obtained at vocabulary size= " +
              str(last_coverage_increase_index))
        print("uncovered words: " + str(uncovered_words))
        types_covered = type_coverage_table[50000]
        tokens_covered = token_coverage_table[50000]
        print("coverage with a 50K words vocabulary:" +
              VocabularyWordCoverageAnalysis.get_type_token_coverage_string(
                  i, types_total_test_set, tokens_total_test_set, types_covered, tokens_covered) + "\n")






def main():

    # if len(sys.argv) != 2:
    #     raise RuntimeError("Error: test_word_frequency_table INPUT_FILE_PATH")
    #
    # input_file_path = sys.argv[1]
    # WordFrequencyTable.test_word_frequency_table(input_file_path)

    if len(sys.argv) != 4:
        raise RuntimeError("Error: vocabulary_word_coverage_analysis "
                           "LANGUAGE_MODEL_TRAINING_FILE_PATH TEST_SET_FILE_PATH "
                           "COLLAPSE_CASING")

    language_model_training_file_path = sys.argv[1]
    test_set_file_path = sys.argv[2]
    collapse_word_casing = Utils.str2bool(sys.argv[3])
    vocabulary_word_coverage_analysis = VocabularyWordCoverageAnalysis.create_vocabulary_word_coverage_analysis(
        language_model_training_file_path, test_set_file_path, collapse_word_casing)
    vocabulary_word_coverage_analysis.make_coverage_for_vocabulary_sizes_table()


if __name__ == "__main__":
    main()

