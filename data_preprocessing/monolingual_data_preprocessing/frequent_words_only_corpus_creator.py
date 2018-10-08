from util.utils import Utils
from data_preprocessing.monolingual_data_preprocessing.word_frequency_table import WordFrequencyTable
import sys


class FrequentWordsOnlyCorpusCreator:
    INFREQUENT_WORD_SYMBOL = "@@@-INFREQUENT-WORD-@@@"

    def __init__(self, input_file_path: str, output_file_path: str,
                 word_frequency_table: WordFrequencyTable,
                 vocabulary_size: int):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.word_frequency_table = word_frequency_table
        self.vocabulary_size = vocabulary_size

    @staticmethod
    def create_frequent_words_only_corpus_creator(
            input_file_path: str, output_file_path: str,
            collapse_word_case_str: str, vocabulary_size_str: str):
        collapse_word_case = Utils.str2bool(collapse_word_case_str)
        vocabulary_size = int(vocabulary_size_str)
        word_frequency_table = WordFrequencyTable.create_word_frequency_table(
            input_file_path, collapse_word_case)
        return FrequentWordsOnlyCorpusCreator(input_file_path, output_file_path,
                                              word_frequency_table, vocabulary_size)

    def make_frequent_words_only_corpus(self):
        with open(self.input_file_path, "r") as input_file:
            with open(self.output_file_path, "w") as output_file:

                for line in input_file:
                    words = line.strip().split()
                    replaced_words = list([])
                    for word in words:
                        word_frequency_rank = self.word_frequency_table.get_word_frequency_rank(word)
                        # print("word_frequency_rank: " + str(word_frequency_rank))
                        if word_frequency_rank >= self.vocabulary_size:
                            replaced_words.append(FrequentWordsOnlyCorpusCreator.INFREQUENT_WORD_SYMBOL)
                        else:
                            replaced_words.append(word)
                    replaced_line = " ".join(replaced_words)
                    output_file.write(replaced_line + "\n")


def main():

    # if len(sys.argv) != 2:
    #     raise RuntimeError("Error: test_word_frequency_table INPUT_FILE_PATH")
    #
    # input_file_path = sys.argv[1]
    # WordFrequencyTable.test_word_frequency_table(input_file_path)

    if len(sys.argv) != 5:
        raise RuntimeError("Error: frequent_words_only_corpus_creator INPUT_FILE_PATH "
                           "OUTPUT_FILE_PATH COLLAPSE_WORD_CASE VOCABULARY_SIZE")

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    collapse_word_case = sys.argv[3]
    vocabulary_size = sys.argv[4]
    frequent_word_corpus_creator = FrequentWordsOnlyCorpusCreator.create_frequent_words_only_corpus_creator(
        input_file_path, output_file_path, collapse_word_case, vocabulary_size)
    frequent_word_corpus_creator.make_frequent_words_only_corpus()


if __name__ == "__main__":
    main()

