import evaluation_metrics.levenshtein_distance as ld
from data_preprocessing.iam_database_preprocessing.iam_examples_dictionary import IamLineInformation
import Levenshtein

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


def character_error_rate_single_output_reference_pair(symbol_list_output: list, symbol_list_reference: list):
    """
    Computes the character error rate for a single [output,reference] pair. This method should
    not be used in the typical case when there are multiple [output,reference] pairs; in this case
    the method "character_error_rate_list_of_output_reference_pairs" is appropriate.
    """
    distance = ld.levenshtein_distance(symbol_list_output, symbol_list_reference)
    print("distance = " + str(distance))
    reference_length = len(symbol_list_reference)
    print("reference_length: " + str(reference_length))
    result = distance / reference_length
    print("result: " + str(result))
    return result


def character_error_rate_single_output_reference_pair_fast(output: str, reference: str):
    """
    Computes the character error rate for a single [output,reference] pair. This method should
    not be used in the typical case when there are multiple [output,reference] pairs; in this case
    the method "character_error_rate_list_of_output_reference_pairs" is appropriate.

    Faster implementation using the python Levehnstein pacakge
    """
    distance = Levenshtein.distance(output, reference)
    print("distance = " + str(distance))
    reference_length = len(reference)
    print("reference_length: " + str(reference_length))
    result = distance / reference_length
    print("result: " + str(result))
    return result


def compute_character_error_rate_for_list_of_output_reference_pairs(
        outputs_as_strings: list, references_as_strings: list, include_word_separators: bool):
    """
    When computing the character error rate for a list of [output,reference] pairs,
    one should sum the Levensthein distances for the pairs and divide the result by the total
    summed reference lengths.
    https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates

    Why do it like this?
    This is opposed to something like computing the average of the metric computed on each
    sentence pair separate, as that would give equal weight to very short and very long sentences,
    even though the longer sentences contribute more character errors in total, which would
    thus be wrong.
    """

    outputs_as_char_lists = create_character_sequences_from_strings(outputs_as_strings, include_word_separators)
    references_as_char_lists = create_character_sequences_from_strings(references_as_strings, include_word_separators)

    total_distance = 0
    total_reference_length = 0

    for symbol_list_output, symbol_list_reference in zip(outputs_as_char_lists, references_as_char_lists):
        distance = ld.levenshtein_distance(symbol_list_output, symbol_list_reference)
        # print("distance = " + str(distance))
        reference_length = len(symbol_list_reference)
        # print("reference_length: " + str(reference_length))
        total_distance += distance
        total_reference_length += reference_length

    result = total_distance / total_reference_length
    # print("result: " + str(result))
    return result


def compute_character_error_rate_for_list_of_output_reference_pairs_fast(
        outputs_as_strings: list, references_as_strings: list, include_word_separators: bool):
    """
    Faster implementation using python Levehnstein package
    """

    total_distance = 0
    total_reference_length = 0

    for output, reference in zip(outputs_as_strings, references_as_strings):
        if not include_word_separators:
            output = output.replace(IamLineInformation.WORD_SEPARATOR_SYMBOL, "")
            reference = output.replace(IamLineInformation.WORD_SEPARATOR_SYMBOL, "")
        distance = Levenshtein.distance(output, reference)
        # print("distance = " + str(distance))
        reference_length = len(reference)
        # print("reference_length: " + str(reference_length))
        total_distance += distance
        total_reference_length += reference_length

    result = total_distance / total_reference_length
    # print("result: " + str(result))
    return result


def test_character_error_rate(char_sequence_one_as_string: str,
                              char_seq_two_as_string: str,
                              expected_character_error_rate: int):
    character_sequence_one = ld.create_character_sequence_from_string(
        char_sequence_one_as_string)
    character_sequence_two = ld.create_character_sequence_from_string(
        char_seq_two_as_string)
    cer = character_error_rate_single_output_reference_pair(character_sequence_one, character_sequence_two)

    if not cer == expected_character_error_rate:
        raise RuntimeError("Error: expected a character error rate of : "
                           + str(expected_character_error_rate) + " between: \"" +
                           char_sequence_one_as_string + "\" and \"" +
                           char_seq_two_as_string + "\" , but got: " + str(cer))


def test_character_error_rate_fast(
        char_sequence_one_as_string: str, char_seq_two_as_string: str, expected_character_error_rate: int):

    cer = character_error_rate_single_output_reference_pair_fast(char_sequence_one_as_string, char_seq_two_as_string)

    if not cer == expected_character_error_rate:
        raise RuntimeError("Error: expected a character error rate of : "
                           + str(expected_character_error_rate) + " between: \"" +
                           char_sequence_one_as_string + "\" and \"" +
                           char_seq_two_as_string + "\" , but got: " + str(cer))


def create_character_sequences_from_strings(strings: list, include_word_separators: bool):
    result = list([])
    for string in strings:
        if include_word_separators:
            string_to_add = string
        else:
            # Remove the word separators from the string
            string_to_add = string.replace(IamLineInformation.WORD_SEPARATOR_SYMBOL, "")
        result.append(ld.create_character_sequence_from_string(string_to_add))
    return result


def test_character_error_rate_list_of_output_reference_pairs(
        outputs_as_strings: list, references_as_strings: list,
        expected_character_error_rate: int):

    cer = compute_character_error_rate_for_list_of_output_reference_pairs(
        outputs_as_strings, references_as_strings, True)

    print("test_character_error_rate_list_of_output_reference_pairs - cer: " + str(cer))

    if not cer == expected_character_error_rate:
        raise RuntimeError("Error: expected a character error rate of : "
                           + str(expected_character_error_rate) + " between: \"" +
                           outputs_as_strings + "\" and \"" +
                           references_as_strings + "\" , but got: " + str(cer))


def test_character_error_rate_list_of_output_reference_pairs_fast(
        outputs_as_strings: list, references_as_strings: list,
        expected_character_error_rate: int):

    cer = compute_character_error_rate_for_list_of_output_reference_pairs_fast(
        outputs_as_strings, references_as_strings, True)

    print("test_character_error_rate_list_of_output_reference_pairs_fast - cer: " + str(cer))

    if not cer == expected_character_error_rate:
        raise RuntimeError("Error: expected a character error rate of : "
                           + str(expected_character_error_rate) + " between: \"" +
                           outputs_as_strings + "\" and \"" +
                           references_as_strings + "\" , but got: " + str(cer))


def test_cer_one():
    # a 	b 	c 	d 	e 	f 	g
    # a 	b 	  	d 	  	f 	h
    char_seq_one_as_string = "abcdefg"
    char_seq_two_as_string = "abdfh"
    test_character_error_rate(char_seq_one_as_string, char_seq_two_as_string, 0.6)


def test_cer_one_fast():
    # a 	b 	c 	d 	e 	f 	g
    # a 	b 	  	d 	  	f 	h
    char_seq_one_as_string = "abcdefg"
    char_seq_two_as_string = "abdfh"
    test_character_error_rate_fast(char_seq_one_as_string, char_seq_two_as_string, 0.6)


def test_cer_two_elements():
    # a 	b 	c 	d 	e 	f 	g
    # a 	b 	  	d 	  	f 	h
    char_seq_one_as_string = "abcdefg"
    char_seq_one_reference_as_string = "abdfh"
    expected_levenshtein_distance_pair_one = 3

    char_seq_two_as_string = "abcdefg"
    char_seq_two_reference_as_string = "ab"
    expected_levenshtein_distance_pair_two = 5

    outputs = list([char_seq_one_as_string, char_seq_two_as_string])
    references = list([char_seq_one_reference_as_string, char_seq_two_reference_as_string])

    expected_total_levenshtein_distance = (expected_levenshtein_distance_pair_one +
                                           expected_levenshtein_distance_pair_two)
    total_summed_reference_lengths = len(char_seq_one_reference_as_string) + len(char_seq_two_reference_as_string)
    expected_result = expected_total_levenshtein_distance / total_summed_reference_lengths
    return outputs, references, expected_result


def test_cer_two():
    outputs, references, expected_result = test_cer_two_elements()
    test_character_error_rate_list_of_output_reference_pairs(outputs, references, expected_result)


def test_cer_two_fast():
    outputs, references, expected_result = test_cer_two_elements()
    test_character_error_rate_list_of_output_reference_pairs_fast(outputs, references, expected_result)


def main():
    test_cer_one()
    test_cer_one_fast()
    test_cer_two()
    test_cer_two_fast()


if __name__ == "__main__":
    main()

