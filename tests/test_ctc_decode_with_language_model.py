import torch
import sys
from language_model.kenlm_interface import KenlmInterface
import ctcdecode
from modules.evaluator import Evaluator

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class TestLanguageModelCreator:
    # HANDWRITING_RECOGNITION_ROOT_DIR = "~/AI/handwriting-recognition/"
    LANGUAGE_MODEL_OUTPUT_DIR_SUFFIX = "tests/language_models/"
    LANGUAGE_MODEL_TRAIN_FILE_NAME = "language_model_train_file.txt"
    LANGUAGE_MODEL_ARPA_FILE_NAME = "test_language_model.arpa"
    LANGUAGE_MODEL_BINARY_FILE_NAME = "test_language_model_binary"
    LANGUAGE_MODEL_TEXT = "Klaas maakt een keuken met een kraan .\n" \
                          "De kraan die hij maakt is aan de kant van de " \
                          "muur .\nDe kraan is duur , maar niet te huur .\n" \
                          "De huur is niet duur. Maar huren van muren " \
                          "kan ook rare kuren sturen .\n" \
                          "Huur een keuken met een kraan .\n"
    LANGUAGE_MODEL_TEXT_TWO = "aaaa bbbb cccc ddd .\n" \
                          "dddd abab cc dad " \
                          "bbbbb .\ncc aa bbbb ddd .\n" \
                          "aa bbb ccc dd. baba " \
                          "aaaa bbbb bb aa cccc .\n" \
                          "abcd ddab dd bbb cccc ddd .\n"\
                          "a b c d .\n"

    def __init__(self, handwriting_recognition_root_dir: str,
                 language_model_output_directory: str, language_model_train_file_name: str,
                 language_model_arpa_file_name: str,
                 language_model_binary_file_name: str):
        self.handwriting_recognition_root_dir = handwriting_recognition_root_dir
        self.language_model_output_directory = language_model_output_directory
        self.language_model_train_file_name = language_model_train_file_name
        self.language_model_arpa_file_name = language_model_arpa_file_name
        self.language_model_binary_file_name = language_model_binary_file_name

    @staticmethod
    def create_test_language_model_creator(handwriting_recognition_root_dir):
        language_model_output_directory = handwriting_recognition_root_dir +\
            TestLanguageModelCreator.LANGUAGE_MODEL_OUTPUT_DIR_SUFFIX
        language_model_train_file_name = TestLanguageModelCreator.LANGUAGE_MODEL_TRAIN_FILE_NAME

        return TestLanguageModelCreator(handwriting_recognition_root_dir,
                                        language_model_output_directory, language_model_train_file_name,
                                        TestLanguageModelCreator.LANGUAGE_MODEL_ARPA_FILE_NAME,
                                        TestLanguageModelCreator.LANGUAGE_MODEL_BINARY_FILE_NAME)

    def get_language_model_train_file_path(self):
        return self.language_model_output_directory + self.language_model_train_file_name

    def get_language_model_arpa_file_path(self):
        return self.language_model_output_directory + self.language_model_arpa_file_name

    def get_language_model_binary_file_path(self):
        return self.language_model_output_directory + self.language_model_binary_file_name

    def create_language_model_train_file(self, language_model_text: str):
        with open(self.get_language_model_train_file_path(), "w") as text_file:
            text_file.write(language_model_text)

    def create_language_model_arpa_file(self, ngram_order: int):
        kenlm_interface = KenlmInterface.create_kenlm_interface(
            self.handwriting_recognition_root_dir)
        kenlm_interface.create_arpa_language_model_for_file(
            ngram_order, self.get_language_model_train_file_path(), self.get_language_model_arpa_file_path())

    def create_language_model_binary_file(self):
        kenlm_interface = KenlmInterface.create_kenlm_interface(
            self.handwriting_recognition_root_dir)
        kenlm_interface.build_binary_language_model_for_file(
            self.get_language_model_arpa_file_path(), self.get_language_model_binary_file_path())
        return self.get_language_model_binary_file_path()


class TestCTCDecodeWithLanguageModel:
    BLANK_SYMBOL = "_"
    BEAM_SIZE = 10
    NONZERO_LANGUAGE_MODEL_WEIGHT = 0.05 #2.15
    WORD_INSERTION_WEIGHT = 0 #0.35

    def __init__(self):
        return

    @staticmethod
    def create_test_vocab_list():
        vocab_list = list(['_', 'a', 'b', 'c', 'd', '.'])
        return vocab_list

    @staticmethod
    def create_test_decoder_with_language_model(handwriting_recognition_root_dir: str,
                                                use_non_zero_language_model_weight: bool):
        vocab_list = TestCTCDecodeWithLanguageModel.create_test_vocab_list()
        language_model_binary_file = create_test_language_model(handwriting_recognition_root_dir)
        # alpha: language model weight
        # beta: word insertion weight
        # See: https://github.com/PaddlePaddle/models/issues/218

        if use_non_zero_language_model_weight:
            language_model_weight = TestCTCDecodeWithLanguageModel.NONZERO_LANGUAGE_MODEL_WEIGHT
            language_model_path = language_model_binary_file
        else:
            language_model_weight = 0
            language_model_path = None

        decoder = ctcdecode.CTCBeamDecoder(vocab_list, model_path= language_model_path,
                                           beam_width=TestCTCDecodeWithLanguageModel.BEAM_SIZE,
                                           alpha=language_model_weight,
                                           beta=TestCTCDecodeWithLanguageModel.WORD_INSERTION_WEIGHT,
                                           blank_id=vocab_list.index(TestCTCDecodeWithLanguageModel.BLANK_SYMBOL),
                                           num_processes=16)
        return decoder, vocab_list

    @ staticmethod
    def create_test_probabilities_and_sequence_lengths():
        # ctcdecode expects batch x seq x label_size
        # probs = torch.zeros(2, 3, 6)
        probs = torch.FloatTensor([
                           [
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0, 0]
                           ],
                           [
                             [0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0.1, 0, 0.9, 0, 0]
                           ],
                          ])
        print("probabilities: " + str(probs))
        print("probabilities.size(): " + str(probs.size()))
        sequence_lengths = torch.LongTensor([3, 3])
        print("sequence_lengths: " + str(sequence_lengths))
        return probs, sequence_lengths

    @ staticmethod
    def create_test_probabilities_and_sequence_lengths_two():
        # ctcdecode expects batch x seq x label_size
        # probs = torch.zeros(2, 4, 6)
        probs = torch.FloatTensor([
                           [
                             [0, 0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0.999, 0, 0, 0.001, 0]
                           ],
                           [
                             [0, 0, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0],
                             [0, 0.999, 0.001, 0, 0, 0]
                           ],
                          ])
        print("probabilities: " + str(probs))
        print("probabilities.size(): " + str(probs.size()))
        sequence_lengths = torch.LongTensor([8, 8])
        print("sequence_lengths: " + str(sequence_lengths))
        return probs, sequence_lengths

    @staticmethod
    def create_test_probabilities_and_sequence_lengths_three():
        # ctcdecode expects batch x seq x label_size
        # probs = torch.zeros(1, 8, 6)
        probs = torch.FloatTensor([
            [
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0.0000, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0.999, 0.001, 0, 0, 0]
            ],
        ])
        print("probabilities: " + str(probs))
        print("probabilities.size(): " + str(probs.size()))
        sequence_lengths = torch.LongTensor([17])
        print("sequence_lengths: " + str(sequence_lengths))
        return probs, sequence_lengths

    @staticmethod
    def test_decoder_with_language_model(probabilities, sequence_lengths,
                                         handwriting_recognition_root_dir,
                                         use_nonzero_language_model_weight: bool):
        decoder, vocab_list = \
            TestCTCDecodeWithLanguageModel.create_test_decoder_with_language_model(handwriting_recognition_root_dir,
                                                                                   use_nonzero_language_model_weight)
        beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(probabilities, sequence_lengths)
        print("output: " + str(beam_results))
        print("scores: " + str(beam_scores))

        output_strings = list([])
        for example_index in range(0, beam_results.size(0)):
            beam_results_sequence = beam_results[example_index][0]
            # print("beam_results_sequence: \"" + str(beam_results_sequence) + "\"")
            output_string = Evaluator.convert_to_string(
                beam_results_sequence, vocab_list, out_seq_len[example_index][0])
            print("output_string: " + str(output_string))
            output_strings.append(output_string)
        return output_strings

    @staticmethod
    def test_decoder_with_language_model_artificial_data(handwriting_recognition_root_dir,
                                                         use_nonzero_language_model_weight: bool):
        probabilities, sequence_lengths = \
            TestCTCDecodeWithLanguageModel.create_test_probabilities_and_sequence_lengths_two()
        return TestCTCDecodeWithLanguageModel.test_decoder_with_language_model(
            probabilities, sequence_lengths, handwriting_recognition_root_dir, use_nonzero_language_model_weight)

    @staticmethod
    def test_decoder_with_language_model_artificial_data_two(handwriting_recognition_root_dir,
                                                             use_nonzero_language_model_weight: bool):
        probabilities, sequence_lengths = \
            TestCTCDecodeWithLanguageModel.create_test_probabilities_and_sequence_lengths_three()
        return TestCTCDecodeWithLanguageModel.test_decoder_with_language_model(
            probabilities, sequence_lengths, handwriting_recognition_root_dir, use_nonzero_language_model_weight)

    @staticmethod
    def test_decoder_with_active_language_model_artificial_data(handwriting_recognition_root_dir):
        outputs = TestCTCDecodeWithLanguageModel.\
            test_decoder_with_language_model_artificial_data(handwriting_recognition_root_dir, True)

        # With active language model, the decoder should just produce the most likely sequences
        # but also make them consistent with the observed sequences used for training the language model.
        # These contain "dddd" and "bbbb", but not "ddda" and "bbba". Hence, "dddd" and "bbbb" should
        # be produced by the decoder, when taking the language model into account.
        expected_outputs = list(["dddd", "bbbb"])

        if not outputs == expected_outputs:
            raise RuntimeError("Error: expected the outputs to be " + str(expected_outputs) + " but got:  " +
                               str(outputs))

    @staticmethod
    def test_decoder_without_active_language_model_artificial_data(handwriting_recognition_root_dir):
        outputs = TestCTCDecodeWithLanguageModel. \
            test_decoder_with_language_model_artificial_data(handwriting_recognition_root_dir, False)
        # Without active language model, the decoder should just produce the most likely sequences
        # which are dda and bbba
        expected_outputs = list(["ddda", "bbba"])

        if not outputs == expected_outputs:
            raise RuntimeError("Error: expected the outputs to be " + str(expected_outputs) + " but got:  " +
                               str(outputs))

    @staticmethod
    def test_decoder_with_active_language_model_artificial_data_two(handwriting_recognition_root_dir):
        outputs = TestCTCDecodeWithLanguageModel. \
            test_decoder_with_language_model_artificial_data_two(handwriting_recognition_root_dir, True)

        # With active language model, the decoder should just produce the most likely sequences
        # but also make them consistent with the observed sequences used for training the language model.
        # This leads to the sequence "dddccbbba" being most likely, even though the "c" in the fourth position
        # actually has zero support in the probabilities
        expected_outputs = list(["dddccbbba"])

        if not outputs == expected_outputs:
            raise RuntimeError("Error: expected the outputs to be " + str(expected_outputs) + " but got:  " +
                               str(outputs))

    @staticmethod
    def test_decoder_without_active_language_model_artificial_data_two(handwriting_recognition_root_dir):
        outputs = TestCTCDecodeWithLanguageModel. \
            test_decoder_with_language_model_artificial_data_two(handwriting_recognition_root_dir, False)
        # Without active language model, the decoder should just produce the most likely sequences
        # which are dda and bbba
        expected_outputs = list(["dddacbbba"])

        if not outputs == expected_outputs:
            raise RuntimeError("Error: expected the outputs to be " + str(expected_outputs) + " but got:  " +
                               str(outputs))


def create_test_language_model(handwriting_recognition_root_dir: str):
    test_language_model_creator = TestLanguageModelCreator.create_test_language_model_creator(
        handwriting_recognition_root_dir)
    test_language_model_creator.create_language_model_train_file(TestLanguageModelCreator.LANGUAGE_MODEL_TEXT_TWO)
    # For a higher order language model a bigger, non-artificial training set may be required.
    test_language_model_creator.create_language_model_arpa_file(2)
    return test_language_model_creator.create_language_model_binary_file()


def main():
    handwriting_recognition_root_dir = sys.argv[1]
    # create_test_language_model(handwriting_recognition_root_dir)
    TestCTCDecodeWithLanguageModel.\
        test_decoder_with_active_language_model_artificial_data(handwriting_recognition_root_dir)
    TestCTCDecodeWithLanguageModel. \
        test_decoder_without_active_language_model_artificial_data(handwriting_recognition_root_dir)
    TestCTCDecodeWithLanguageModel. \
        test_decoder_with_active_language_model_artificial_data_two(handwriting_recognition_root_dir)
    TestCTCDecodeWithLanguageModel. \
        test_decoder_without_active_language_model_artificial_data_two(handwriting_recognition_root_dir)


if __name__ == "__main__":
    main()


