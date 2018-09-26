from abc import abstractmethod
import sys
from collections import OrderedDict

class WordCasingGrouping:

    @abstractmethod
    def get_word_group(self, word: str):
        raise RuntimeError("Not implemented")


class LowerCaseWordGrouping(WordCasingGrouping):

    def get_word_group(self, word: str):
        return word.lower()


class KeepCaseWordGrouping(WordCasingGrouping):

    def get_word_group(self, word: str):
        return word


class WordFrequencyTable:

    def __init__(self, word_group_frequency_table: dict,
                 word_case_grouping: WordCasingGrouping):
        self.word_group_frequency_table = word_group_frequency_table
        self.word_case_grouping = word_case_grouping
        self.word_groups_sorted_by_count = None
        self.word_group_frequency_rank_table = None

    @staticmethod
    def create_word_frequency_table(input_file_path: str, collapse_word_case: bool):
        if collapse_word_case:
            word_case_grouping = LowerCaseWordGrouping()
        else:
            word_case_grouping = KeepCaseWordGrouping()
        word_frequency_table = WordFrequencyTable(dict([]), word_case_grouping)

        with open(input_file_path, 'r') as file:
            for line in file:
                for word in line.split():
                    word_frequency_table.increase_word_group_count(word)

        word_frequency_table.compute_word_groups_sorted_by_count_and_word_frequency_rank_table()

        return word_frequency_table

    def increase_word_group_count(self, word):
        word_group = self.word_case_grouping.get_word_group(word)

        if word_group in self.word_group_frequency_table:
            current_count = self.word_group_frequency_table[word_group]
            new_count = current_count + 1
            self.word_group_frequency_table[word_group] = new_count
        else:
            self.word_group_frequency_table[word_group] = 1

    def compute_word_groups_sorted_by_count_and_word_frequency_rank_table(self):
        self.word_groups_sorted_by_count = sorted(self.word_group_frequency_table.items(), key=lambda kv: kv[1])
        self.word_groups_sorted_by_count.reverse()

        self.word_group_frequency_rank_table = OrderedDict()
        for i, word_frequency_tuple in enumerate(self.word_groups_sorted_by_count):
            self.word_group_frequency_rank_table[word_frequency_tuple[0]] = i

    @staticmethod
    def test_word_frequency_table(input_file_path: str):
        word_frequency_table = WordFrequencyTable.\
            create_word_frequency_table(input_file_path, False)
        # print("\nword_frequency_table.word_groups_sorted_by_count: "
        #       + str(word_frequency_table.word_groups_sorted_by_count))
        print("\nword_frequency_table.word_group_frequency_rank_table: "
              + str(word_frequency_table.word_group_frequency_rank_table))

def main():

    if len(sys.argv) != 2:
        raise RuntimeError("Error: test_word_frequency_table INPUT_FILE_PATH")

    input_file_path = sys.argv[1]
    WordFrequencyTable.test_word_frequency_table(input_file_path)


if __name__ == "__main__":
    main()

