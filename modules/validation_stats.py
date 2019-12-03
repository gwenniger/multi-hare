
"""
Validation stats class. Stores (exact match) accuracy, and perhaps in the future other
validation metrics
"""

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"

class ValidationStats:

    def __init__(self, number_of_examples, number_correct, character_error_rate: float, word_error_rate: float):
        self.number_of_examples = number_of_examples
        self.number_correct = number_correct
        self.character_error_rate = character_error_rate
        self.word_error_rate = word_error_rate

    def get_accuracy(self):
        accuracy = float(100 * self.number_correct) / self.number_of_examples
        return accuracy

    def get_character_error_rate(self):
        return self.character_error_rate

    def get_word_error_rate(self):
        return self.word_error_rate
