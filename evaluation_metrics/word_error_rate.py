import evaluation_metrics.levenshtein_distance as ld
from data_preprocessing.iam_database_preprocessing.iam_examples_dictionary import IamLineInformation

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


"""
The character error rate (CER is depfined based upon the
Levenshtein distance, the total of insertions, substitutions and deletions
(i + s + d).


https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates
"
The character error rate is defined in a similar way as:

CER = (i + s + d) / n

but using the total number n of characters and the minimal number of character insertions i,
substitutions s an deletions d required to transform the reference text into the OCR output.
"

   but using the total number n of characters = > n characters of the reference
   (otherwise it wouldn't make much sense)
"""


def word_error_rate_single_output_reference_pair(symbol_list_output: list, symbol_list_reference: list):
    """
    Computes the character error rate for a single [output,reference] pair. This method should
    not be used in the typical case when there are multiple [output,reference] pairs; in this case
    the method "character_error_rate_list_of_output_reference_pairs" is appropriate.
    """
    distance = ld.levenshtein_distance(symbol_list_output, symbol_list_reference)
    # print("distance = " + str(distance))
    reference_length = len(symbol_list_reference)
    # print("reference_length: " + str(reference_length))
    result = distance / reference_length
    # print("result: " + str(result))
    # Multiply by 100 to make it a percentage
    result *= 100
    return result


def compute_word_error_rate_for_list_of_output_reference_pairs(outputs_as_strings: list,
                                                               references_as_strings: list):
    """
    When computing the word error rate for a list of [output,reference] pairs,
    one should sum the Levensthein distances for the pairs and divide the result by the total
    summed reference lengths.
    https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates

    Why do it like this?
    This is opposed to something like computing the average of the metric computed on each
    sentence pair separate, as that would give equal weight to very short and very long sentences,
    even though the longer sentences contribute more character errors in total, which would
    thus be wrong.
    """

    outputs_as_word_lists = create_word_sequences_from_strings(outputs_as_strings)
    references_as_word_lists = create_word_sequences_from_strings(references_as_strings)

    total_distance = 0
    total_reference_length = 0

    for symbol_list_output, symbol_list_reference in zip(outputs_as_word_lists, references_as_word_lists):
        distance = ld.levenshtein_distance(symbol_list_output, symbol_list_reference)
        print("compute_word_error_rate_for_list_of_output_reference_pairs - distance = " + str(distance))
        reference_length = len(symbol_list_reference)
        # print("reference_length: " + str(reference_length))
        total_distance += distance
        total_reference_length += reference_length

    print("compute_word_error_rate_for_list_of_output_reference_pairs - total_distance: " +
          str(total_distance) + " total reference length: " + str(total_reference_length))

    result = total_distance / total_reference_length
    # print("result: " + str(result))
    # Multiply by 100 to make it a percentage
    result *= 100
    return result


def test_word_error_rate(word_sequence_one_as_string: str,
                         word_seq_two_as_string: str,
                         expected_character_error_rate: int):
    word_sequence_one = create_word_sequence_from_string(
        word_sequence_one_as_string)
    word_sequence_two = create_word_sequence_from_string(
        word_seq_two_as_string)
    wer = word_error_rate_single_output_reference_pair(word_sequence_one, word_sequence_two)

    if not wer == expected_character_error_rate:
        raise RuntimeError("Error: expected a word error rate of : "
                           + str(expected_character_error_rate) + " between: \"" +
                           word_sequence_one_as_string + "\" and \"" +
                           word_seq_two_as_string + "\" , but got: " + str(wer))


def create_word_sequence_from_string(strings: list):
    return strings.split(IamLineInformation.WORD_SEPARATOR_SYMBOL)


def create_word_sequences_from_strings(strings: list):
    result = list([])
    for string in strings:
        result.append(create_word_sequence_from_string(string))
    return result


def test_word_error_rate_list_of_output_reference_pairs(
        outputs_as_strings: list, references_as_strings: list,
        expected_word_error_rate: int):

    cer = compute_word_error_rate_for_list_of_output_reference_pairs(
        outputs_as_strings, references_as_strings)

    if not cer == expected_word_error_rate:
        raise RuntimeError("Error: expected a word error rate of : "
                           + str(expected_word_error_rate) + " between: \"" +
                           str(outputs_as_strings) + "\" and \"" +
                           str(references_as_strings) + "\" , but got: " + str(cer))


def test_wer_one():
    # a 	b 	c 	d 	e 	f 	g
    # a 	b 	  	d 	  	f 	h
    char_seq_one_as_string = "aa|bb|cc|dd|ee|ff|g"
    char_seq_two_as_string = "aa|bb|dd|ff|h"
    test_word_error_rate(char_seq_one_as_string, char_seq_two_as_string, 0.6)


def test_wer_two():
    # a 	b 	c 	d 	e 	f 	g
    # a 	b 	  	d 	  	f 	h
    char_seq_one_as_string = "aa|bb|cc|d|e|f|g"
    char_seq_one_reference_as_string = "aa|bb|d|f|h"
    expected_levenshtein_distance_pair_one = 3

    char_seq_two_as_string = "aa|bb|cc|d|e|f|g"
    char_seq_two_reference_as_string = "aa|bb"
    expected_levenshtein_distance_pair_two = 5

    outputs = list([char_seq_one_as_string, char_seq_two_as_string])
    references = list([char_seq_one_reference_as_string, char_seq_two_reference_as_string])

    expected_total_levenshtein_distance = (expected_levenshtein_distance_pair_one +
                                           expected_levenshtein_distance_pair_two)
    total_summed_reference_lengths = \
        len(char_seq_one_reference_as_string.split(IamLineInformation.WORD_SEPARATOR_SYMBOL)) + \
        len(char_seq_two_reference_as_string.split(IamLineInformation.WORD_SEPARATOR_SYMBOL))
    expected_result = expected_total_levenshtein_distance / total_summed_reference_lengths
    test_word_error_rate_list_of_output_reference_pairs(outputs, references, expected_result)


def main():
    test_wer_one()
    test_wer_two()


if __name__ == "__main__":
    main()

