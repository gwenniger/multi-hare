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


class Evaluator:

    # Note that if seq_len=0 then the result will always be the empty String
    @staticmethod
    def convert_to_string(tokens, vocab, seq_len):
        # print("convert_to_string - tokens: " + str(tokens))
        # print("convert_to_string - vocab: " + str(vocab))
        # print("convert_to_string - seq_len: " + str(seq_len))
        result = ''.join([vocab[x] for x in tokens[0:seq_len]])
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
    def evaluate_mdrnn(test_loader, multi_dimensional_rnn, device,
                       vocab_list: list, blank_symbol: str, horizontal_reduction_factor: int,
                       image_input_is_unsigned_int: bool, minimize_horizontal_padding: bool):

        correct = 0
        total = 0

        for data in test_loader:
            inputs, labels = data

            if Utils.use_cuda():
                labels = labels.to(device)

                if minimize_horizontal_padding:
                    inputs = Utils.move_tensor_list_to_device(inputs, device)
                else:
                    inputs = inputs.to(device)

            # If the image input comes in the form of unsigned ints, they need to
            # be converted to floats (after moving to GPU, i.e. directly on GPU
            # which is faster)
            if image_input_is_unsigned_int:
                Trainer.check_inputs_is_right_type(inputs, minimize_horizontal_padding)
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

                print(">>> evaluate_mdrnn  - outputs.size: " + str(outputs.size()))
                print(">>> evaluate_mdrnn  - probabilities.size: " + str(probabilities.size()))

                beam_size = 20
                print(">>> evaluate_mdrnn  - len(vocab_list): " + str(len(vocab_list)))
                decoder = ctcdecode.CTCBeamDecoder(vocab_list, beam_width=beam_size,
                                                   blank_id=vocab_list.index(blank_symbol),
                                                   num_processes=16)
                label_sizes = WarpCTCLossInterface. \
                    create_sequence_lengths_specification_tensor_different_lengths(labels)

                sequence_lengths = WarpCTCLossInterface.\
                    create_probabilities_lengths_specification_tensor_different_lengths(
                        labels, horizontal_reduction_factor, probabilities)
                # print(">>> evaluate_mdrnn  -  sequence lengths: " + str(sequence_lengths))
                # print("probabilities.data.size(): " + str(probabilities.data.size()))
                beam_results, beam_scores, timesteps, out_seq_len = \
                    decoder.decode(probabilities.data, sequence_lengths)

                # print(">>> evaluate_mdrnn  - beam_results: " + str(beam_results))

                total += labels.size(0)

                output_strings = list([])
                reference_labels_strings = list([])

                for example_index in range(0, beam_results.size(0)):
                    beam_results_sequence = beam_results[example_index][0]
                    # print("beam_results_sequence: \"" + str(beam_results_sequence) + "\"")
                    output_string = Evaluator.convert_to_string(
                        beam_results_sequence, vocab_list, out_seq_len[example_index][0])
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

        print("Character Error Rate (CER) of the network on the {} test inputs, "
              "including word separators: {:.3f}  CER".format(
                total_examples, cer_including_word_separators))
        print("Character Error Rate (CER) of the network on the {} test inputs, "
              "excluding word separators: {:.3f}  CER".format(
                total_examples, cer_excluding_word_separators))
        print("Word Error Rate (WER) of the network on the {} test inputs: {:.3f}  WER".format(
            total_examples, wer))

        return validation_stats
