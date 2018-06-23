
"""
Validation stats class. Stores (exact match) accuracy, and perhaps in the future other
validation metrics
"""


class ValidationStats:

    def __init__(self, number_of_examples, number_correct):
        self.number_of_examples = number_of_examples
        self.number_correct = number_correct

    def accuracy(self):
        accuracy = float(100 * self.number_correct) / self.number_of_examples
        return accuracy
