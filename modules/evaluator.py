from util.utils import Utils
import torch
from ctc_loss.warp_ctc_loss_interface import WarpCTCLossInterface
from modules.trainer import Trainer
import ctcdecode
from data_preprocessing.iam_database_preprocessing.iam_dataset import IamLinesDataset
from modules.validation_stats import ValidationStats
from modules.network_to_softmax_network import NetworkToSoftMaxNetwork
import evaluation_metrics.character_error_rate
import evaluation_metrics.word_error_rate
from util.nvidia_smi_memory_usage_statistics_collector import GpuMemoryUsageStatistics
import re
import os
import util.timing

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class LanguageModelParameters:

    def __init__(self, language_model_file_path: str,
                 language_model_weight: float,
                 word_insertion_penalty: float):
        self.language_model_file_path = language_model_file_path
        self.language_model_weight = language_model_weight
        self.word_insertion_penalty = word_insertion_penalty


class EpochStatistics:
    def __init__(self, total_examples: int,
                 average_loss_per_minibatch: float, time_start, time_end,
                 gpus_memory_usage_statistics: dict):
        self.total_examples = total_examples
        self.average_loss_per_minibatch = average_loss_per_minibatch
        self.time_start = time_start
        self.time_end = time_end
        self.gpus_memory_usage_statistics = gpus_memory_usage_statistics

    def time_passed_in_seconds(self):
        return util.timing.seconds_since_static(self.time_start, self.time_end)

    def get_number_of_used_gpus(self):
        return len(self.gpus_memory_usage_statistics.keys())

    def get_number_of_examples(self):
        return self.total_examples

    def get_number_of_examples_per_second(self):
        return float(self.get_number_of_examples()) / self.time_passed_in_seconds()


class Evaluator:
    WORD_SEPARATOR_SYMBOL = "|"
    BEAM_SIZE = 1000

    # Note that if seq_len=0 then the result will always be the empty String
    @staticmethod
    def convert_to_string(tokens, vocab, seq_len, use_language_model_in_decoder: bool):
        # print("convert_to_string - tokens: " + str(tokens))
        # print("convert_to_string - vocab: " + str(vocab))
        # print("convert_to_string - seq_len: " + str(seq_len))

        result = ''.join([vocab[x] for x in tokens[0:seq_len]])

        # Somehow the decoder with language model sometimes produces output
        # that ends with a word separator. This can never be right, so the last 
        # word separator is deterministically removed in this case. 
        # TODO: This is of course a hack, though an acceptable one, but it should rather be fixed in the decoder
        if result.endswith(Evaluator.WORD_SEPARATOR_SYMBOL):
            result = result[0:len(result) - 1]	

        # print("convert_to_string - result: " + str(result))
        return result

    @staticmethod
    def convert_labels_tensor_to_string(labels: torch.Tensor, vocab_list: list, blank_symbol):

        # Check that the first label of vocab_list is indeed the blank label,
        # otherwise the results will be incorrect
        if not vocab_list[0] == blank_symbol:
            raise RuntimeError("Error: convert_labels_tensor_to_string - " +
                               "requires the first label of vocab_list (" +
                               str(vocab_list) + ") to be the blank symbol " +
                               " but was: " + str(vocab_list[0]))

        labels_as_list = labels.data.tolist()
        result = ""
        for i in range(0, labels.size(0)):
            index = labels_as_list[i]
            # print("convert_labels_tensor_to_string - index: " + str(index))
            result = result + str(vocab_list[index])
        return result

    @staticmethod
    def create_decoder(vocab_list: list, cutoff_top_n: int,
                       beam_size: int,
                       blank_symbol,
                       language_model_parameters: LanguageModelParameters):
        """

        :param vocab_list:
        :param beam_size:
        :param cutoff_top_n:  A parameter that limits the number of vocabulary
                              candidates that are kept by the decoder.
        :param blank_symbol:
        :param language_model_parameters:
        :return:
        """
        if language_model_parameters is not None:

            print("Creating decoder with language model loaded from " +
                  str(language_model_parameters.language_model_file_path))

            decoder = ctcdecode.\
                CTCBeamDecoder(
                    vocab_list, model_path=language_model_parameters.language_model_file_path,
                    cutoff_top_n=cutoff_top_n,
                    beam_width=beam_size, alpha=language_model_parameters.language_model_weight,
                    beta=language_model_parameters.word_insertion_penalty,
                    blank_id=vocab_list.index(blank_symbol),
                    space_symbol=Evaluator.WORD_SEPARATOR_SYMBOL,
                    num_processes=16)
        else:

            decoder = ctcdecode.CTCBeamDecoder(vocab_list, cutoff_top_n=cutoff_top_n,
                                               beam_width=beam_size,
                                               blank_id=vocab_list.index(blank_symbol),
                                               space_symbol=Evaluator.WORD_SEPARATOR_SYMBOL,
                                               num_processes=16)
        return decoder

    @staticmethod
    def append_preceding_word_separator_to_probabilities(probabilities: torch.Tensor,
                                                         vocab_list: list, word_separator_symbol: str):
        """
        The goal of this method is to add artificial probabilities 1 for the word separator symbol
        as an extra symbol probabilities column in probabilities. This is to allow the decoder
        to find a word separator symbol, which is not actually in the original probabilities, but which is
        needed to allow the language model to be trained with a word separator symbol pre-pended at the
        beginning of each word.

        """
        print("probabilities.size(): " + str(probabilities.size()))
        batch_size = probabilities.size(0)
        number_of_symbols_including_blank = probabilities.size(2)
        extra_probabilities_slice = torch.zeros([batch_size, number_of_symbols_including_blank],
                                                device=probabilities.get_device())
        word_separator_index = vocab_list.index(word_separator_symbol)
        extra_probabilities_slice[:, word_separator_index] = 1
        extra_probabilities_slice = extra_probabilities_slice.unsqueeze(1)
        print("extra_probabilities_slice: " + str(extra_probabilities_slice))
        probabilities_with_artificial_preceding_word_separator = \
            torch.cat((extra_probabilities_slice, probabilities), 1)
        return probabilities_with_artificial_preceding_word_separator

    @staticmethod
    def increase_sequence_lengths_by_one(sequence_lengths: torch.Tensor):
        print("sequence_lengths: " + str(sequence_lengths))
        sequence_lengths = sequence_lengths + torch.ones_like(sequence_lengths)
        print("sequence_lengths after: " + str(sequence_lengths))
        return sequence_lengths

    @staticmethod
    def gpu_index_prefix(gpu_index):
        return "gpu_" + str(gpu_index) + "_"

    @staticmethod
    def epoch_statistics_header_part(epoch_statistics: EpochStatistics):
        result = ""
        for gpu_index in epoch_statistics.gpus_memory_usage_statistics.keys():
            # gpu_memory_usage_statistics = epoch_statistics.gpus_memory_usage_statistics[gpu_index]
            result += Evaluator.gpu_index_prefix(gpu_index) + "min_memory_usage" + "," + \
                      Evaluator.gpu_index_prefix(gpu_index) + "max_memory_usage" + "," + \
                      Evaluator.gpu_index_prefix(gpu_index) + "mean_memory_usage" + "," + \
                      Evaluator.gpu_index_prefix(gpu_index) + "stdev_memory_usage" + ","
        return result

    @staticmethod
    def reduce_decimals(input_number: float, decimals: int):
        format_string = "{0:." + str(decimals) + "f}"
        # print("format_string: " + str(format_string))
        return float(format_string.format(input_number))

    @staticmethod
    def epoch_statistics_line_part(epoch_statistics: EpochStatistics):
        result = ""
        for gpu_index in epoch_statistics.gpus_memory_usage_statistics.keys():
            gpu_memory_usage_statistics = epoch_statistics.gpus_memory_usage_statistics[gpu_index]
            result += str(gpu_memory_usage_statistics.get_min_memory_usage()) + "," + \
                str(gpu_memory_usage_statistics.get_max_memory_usage()) + "," + \
                str(Evaluator.reduce_decimals(gpu_memory_usage_statistics.get_mean_memory_usage(), 2)) + "," + \
                str(Evaluator.reduce_decimals(gpu_memory_usage_statistics.get_stdev_memory_usage(), 2)) + ","
        return result

    @staticmethod
    def score_table_header(dev_set_size: int, epoch_statistics: EpochStatistics):
        result = "Scores on the development set of size: " + str(dev_set_size) +"\n"
        result += "\nepoch_number,total_correct,accuracy,CER_including_word_separators[%]," \
                  "CER_excluding_word_separators[%],WER[%],Average_training_CTC_loss_per_minibatch," \
                  "time_in_seconds,number_of_examples,examples_per_second," + \
                  Evaluator.epoch_statistics_header_part(epoch_statistics) + "\n"
        return result

    @staticmethod
    def score_table_line(epoch_number: int, total_correct: int, accuracy: float,
                         cer_including_word_separators: float, cer_excluding_word_separator: float,
                         wer: float, epoch_statistics: EpochStatistics):
        result = str(epoch_number) + "," + str(total_correct) + "," + str(accuracy) + "," +\
                 str(cer_including_word_separators) + "," + str(cer_excluding_word_separator) + "," + str(wer) +\
                 "," + str(epoch_statistics.average_loss_per_minibatch) + "," +\
                 str(epoch_statistics.time_passed_in_seconds()) + "," +\
                 str(epoch_statistics.get_number_of_examples()) + "," +\
                 str(epoch_statistics.get_number_of_examples_per_second()) + "," +\
                 Evaluator.epoch_statistics_line_part(epoch_statistics)
        return result

    @staticmethod
    def evaluate_mdrnn(test_loader, multi_dimensional_rnn, device,
                       vocab_list: list, blank_symbol: str, horizontal_reduction_factor: int,
                       image_input_is_unsigned_int: bool, input_is_list: bool,
                       language_model_parameters: LanguageModelParameters,
                       save_score_table_file_path: str, epoch_number: int, epoch_statistics: EpochStatistics):

        correct = 0
        total = 0

        output_strings = list([])
        reference_labels_strings = list([])

        for data in test_loader:
            inputs, labels = data

            if Utils.use_cuda():
                labels = labels.to(device)

                if input_is_list:
                    inputs = Utils.move_tensor_list_to_device(inputs, device)
                else:
                    inputs = inputs.to(device)

            # If the image input comes in the form of unsigned ints, they need to
            # be converted to floats (after moving to GPU, i.e. directly on GPU
            # which is faster)
            if image_input_is_unsigned_int:
                Trainer.check_inputs_is_right_type(inputs, input_is_list)
                inputs = IamLinesDataset.convert_unsigned_int_image_tensor_or_list_to_float_image_tensor_or_list(inputs)

            # https://github.com/pytorch/pytorch/issues/235
            # Running the evaluation without computing gradients is the recommended way
            # since this saves time, and more importantly, memory
            with torch.no_grad():

                # outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
                max_input_width = NetworkToSoftMaxNetwork.get_max_input_width(inputs)
                outputs = multi_dimensional_rnn(inputs, max_input_width)

                probabilities_sum_to_one_dimension = 2
                # Outputs is the output of the linear layer which is the input to warp_ctc
                # But to get probabilities for the decoder, the softmax function needs to
                # be applied to the outputs
                probabilities = torch.nn.functional. \
                    softmax(outputs, probabilities_sum_to_one_dimension)

                # No longer necessary with fixed word separator specification in decoder
                # and normal language model
                # probabilities = Evaluator.append_preceding_word_separator_to_probabilities(
                #    probabilities, vocab_list, Evaluator.WORD_SEPARATOR_SYMBOL)

                print(">>> evaluate_mdrnn  - outputs.size: " + str(outputs.size()))
                print(">>> evaluate_mdrnn  - probabilities.size: " + str(probabilities.size()))

                # beam_size = 20   # This is the problem perhaps...
                # beam_size = 100  # The normal default is 100
                beam_size = Evaluator.BEAM_SIZE  # Larger value to see if it further improves results
                # This value specifies the number of (character) probabilities kept in the
                # decoder. If it is set equal or larger to the number of characters in the
                # vocabulary, no pruning is done for it
                cutoff_top_n = len(vocab_list)  # No pruning for this parameter
                print(">>> evaluate_mdrnn  - len(vocab_list): " + str(len(vocab_list)))
                decoder = Evaluator.create_decoder(vocab_list,  cutoff_top_n, beam_size,
                                                   blank_symbol,
                                                   language_model_parameters)
                label_sizes = WarpCTCLossInterface. \
                    create_sequence_lengths_specification_tensor_different_lengths(labels)

                sequence_lengths = WarpCTCLossInterface.\
                    create_probabilities_lengths_specification_tensor_different_lengths(
                        labels, horizontal_reduction_factor, probabilities)
                sequence_lengths = Evaluator.increase_sequence_lengths_by_one(sequence_lengths)
                # print(">>> evaluate_mdrnn  -  sequence lengths: " + str(sequence_lengths))
                # print("probabilities.data.size(): " + str(probabilities.data.size()))
                beam_results, beam_scores, timesteps, out_seq_len = \
                    decoder.decode(probabilities.data, sequence_lengths)

                # print(">>> evaluate_mdrnn  - beam_results: " + str(beam_results))

                total += labels.size(0)

                for example_index in range(0, beam_results.size(0)):
                    beam_results_sequence = beam_results[example_index][0]
                    # print("beam_results_sequence: \"" + str(beam_results_sequence) + "\"")
                    use_language_model_in_decoder = language_model_parameters is not None
                    output_string = Evaluator.convert_to_string(
                        beam_results_sequence, vocab_list, out_seq_len[example_index][0],
                        use_language_model_in_decoder)
                    example_labels_with_padding = labels[example_index]
                    # Extract the real example labels, removing the padding labels
                    reference_labels = example_labels_with_padding[0:label_sizes[example_index]]

                    # print(">>> evaluate_mdrnn  - reference_labels: " + str(reference_labels))
                    reference_labels_string = Evaluator.convert_labels_tensor_to_string(
                        reference_labels, vocab_list, blank_symbol)

                    if reference_labels_string == output_string:
                        # print("Yaaaaah, got one correct!!!")
                        correct += 1
                        correct_string = "correct"
                    else:
                        correct_string = "wrong"

                    print(">>> evaluate_mdrnn  - output: \"" + output_string + "\" " +
                          "\nreference: \"" + reference_labels_string + "\" --- "
                          + correct_string)

                    output_strings.append(output_string)
                    reference_labels_strings.append(reference_labels_string)

            # correct += (predicted == labels).sum()

        cer_including_word_separators = evaluation_metrics.character_error_rate. \
            compute_character_error_rate_for_list_of_output_reference_pairs_fast(
                output_strings, reference_labels_strings, True)

        cer_excluding_word_separators = evaluation_metrics.character_error_rate. \
            compute_character_error_rate_for_list_of_output_reference_pairs_fast(
                output_strings, reference_labels_strings, False)

        wer = evaluation_metrics.word_error_rate. \
            compute_word_error_rate_for_list_of_output_reference_pairs(
                output_strings, reference_labels_strings)

        total_examples = len(test_loader.dataset)
        validation_stats = ValidationStats(total_examples, correct, cer_excluding_word_separators, wer)
        # https://stackoverflow.com/questions/3395138/using-multiple-arguments-for-string-formatting-in-python-e-g-s-s
        print("Accuracy of the network on the {} test inputs: {:.2f} % accuracy".format(
            total_examples, validation_stats.get_accuracy()))

        print("Character Error Rate (CER)[%] of the network on the {} test inputs, "
              "including word separators: {:.3f}  CER".format(
                total_examples, cer_including_word_separators))
        print("Character Error Rate (CER)[%] of the network on the {} test inputs, "
              "excluding word separators: {:.3f}  CER".format(
                total_examples, cer_excluding_word_separators))
        print("Word Error Rate (WER)[%] of the network on the {} test inputs: {:.3f}  WER".format(
            total_examples, wer))

        if save_score_table_file_path is not None:
            score_file_existed = os.path.exists(save_score_table_file_path)
            # Opens the file in append-mode, create if it doesn't exists
            with open(save_score_table_file_path, "a") as scores_table_file:
                if not score_file_existed:
                    scores_table_file.write(Evaluator.score_table_header(total_examples, epoch_statistics))
                scores_table_file.write(Evaluator.score_table_line(epoch_number, correct,
                                                                   validation_stats.get_accuracy(),
                                                                   cer_including_word_separators,
                                                                   cer_excluding_word_separators,
                                                                   wer,
                                                                   epoch_statistics) + "\n")

        return validation_stats
