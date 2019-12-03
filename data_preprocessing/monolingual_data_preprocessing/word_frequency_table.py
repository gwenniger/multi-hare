from abc import abstractmethod
import sys
from collections import OrderedDict
from util.utils import Utils

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


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
    def create_word_frequency_table(input_file_path: str, collapse_word_case: bool,
                                    split_symbol: str = None):
        if collapse_word_case:
            print(">>>collapsing word case")
            word_case_grouping = LowerCaseWordGrouping()
        else:
            word_case_grouping = KeepCaseWordGrouping()
        word_frequency_table = WordFrequencyTable(dict([]), word_case_grouping)

        with open(input_file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if split_symbol is not None:
                    words = stripped_line.split(split_symbol)
                else:
                    words = stripped_line.split()

                for word in words:
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

        print("Number of word-groups:" + str(len(self.word_groups_sorted_by_count)))

        self.word_group_frequency_rank_table = OrderedDict()
        for i, word_frequency_tuple in enumerate(self.word_groups_sorted_by_count):
            self.word_group_frequency_rank_table[word_frequency_tuple[0]] = i

    def get_word_frequency(self, word: str):
        word_group = self.word_case_grouping.get_word_group(word)
        return self.word_group_frequency_table[word_group]

    def get_word_frequency_rank(self, word: str):
        word_group = self.word_case_grouping.get_word_group(word)
        return self.word_group_frequency_rank_table[word_group]

    def get_total_word_count(self):
        total_count = 0
        for count in self.word_group_frequency_table.values():
            total_count += count
        return total_count



