import torch
import torch.nn
import time
from util.utils import Utils
from modules.size_two_dimensional import SizeTwoDimensional
import util.timing
import util.tensor_utils
from data_preprocessing.iam_database_preprocessing.iam_dataset import IamLinesDataset
import sys
from modules.gradient_clipping import GradientClipping
import torch.nn
from modules.validation_stats import ValidationStats
from modules.optim import Optim
import modules.find_bad_gradients
from graphviz import render


class ModelProperties:

    def __init__(self, image_input_is_unsigned_int, width_reduction_factor: int):
        self.image_input_is_unsigned_int = image_input_is_unsigned_int
        self.width_reduction_factor = width_reduction_factor


class Trainer:

    def __init__(self, model, optimizer: Optim,
                 warp_ctc_loss_interface,
                 model_properties: ModelProperties):
        self.model = model
        self.optimizer = optimizer
        self.warp_ctc_loss_interface = warp_ctc_loss_interface
        self.model_properties = model_properties
        return

    # Check that the inputs are of ByteTensor (uint8) type
    # Reading the data and preserving it in this type is sort of tricky, so it is
    # best to check that the inputs is indeed of the expected type
    @staticmethod
    def check_inputs_is_right_type(inputs, minimize_horizontal_padding: bool):
        if Utils.use_cuda():
            expected_type_instance = torch.cuda.ByteTensor()
        else:
            expected_type_instance = torch.ByteTensor()

        # If minimize_horizontal_padding is used, the inputs will be a list
        # in this case just check the first element of the list
        if minimize_horizontal_padding:
            item_to_compare = inputs[0]
        else:
            item_to_compare = inputs

        if item_to_compare.type() != expected_type_instance.type():
            raise RuntimeError("Error: expected a " + str(expected_type_instance.type()) + " type image tensor" +
                               " but got : " + str(item_to_compare.type()))

    @staticmethod
    def check_there_are_no_zero_labels(labels, minimize_horizontal_padding: bool):
        # The format expected by warp_ctc, which reserves the 0 label for blanks
        if not minimize_horizontal_padding:
            number_of_zeros = util.tensor_utils.TensorUtils.number_of_zeros(labels)
        else:
            # If minimize_horizontal_padding is used labels will be a list
            # rather than a tensor
            number_of_zeros = 0
            for label_tensor in labels:
                number_of_zeros = util.tensor_utils.TensorUtils.number_of_zeros(label_tensor)

        # Check that labels indeed start from 1 as required
        if number_of_zeros != 0:
            raise RuntimeError("Error: labels tensor contains zeros, which is " +
                               " not allowed, since 0 is reserved for blanks")

    def train_one_epoch(self, train_loader, epoch: int, start: int, batch_size,
                        device, inputs_is_list: bool, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging
            train_loader: the train loader,
            start: time in seconds training started

        """
        # if isinstance(self.model, torch.nn.DataParallel):
        #     device = self.model.module.get_device()
        # else:
        #     device = self.model.get_device()

        num_gradient_corrections = 0
        gradient_norms_sum = 0
        running_loss = 0.0
        time_start = time.time()
        for i, data in enumerate(train_loader, 0):

            time_start_batch = time.time()

            # get the inputs
            inputs, labels = data

            Trainer.check_there_are_no_zero_labels(labels, inputs_is_list)

            # If minimize_horizontal_padding is used, inputs will be a list
            if Utils.use_cuda():
                if not inputs_is_list:
                    inputs = inputs.to(device)
                else:
                    inputs = Utils.move_tensor_list_to_device(inputs, device)

            # If the image input comes in the form of unsigned ints, they need to
            # be converted to floats (after moving to GPU, i.e. directly on GPU
            # which is faster)
            if self.model_properties.image_input_is_unsigned_int:
                Trainer.check_inputs_is_right_type(inputs, inputs_is_list)
                inputs = IamLinesDataset.convert_unsigned_int_image_tensor_or_list_to_float_image_tensor_or_list(inputs)

            if inputs_is_list:
                for element in inputs:
                    element.requires_grad_(True)
            else:
                # Set requires_grad(True) directly and only for the input
                inputs.requires_grad_(True)

            # wrap them in Variable
            # labels = Variable(labels)  # Labels need no gradient apparently
            # if Utils.use_cuda():

            # Labels must remain on CPU for warp-ctc loss
            # labels = labels.to(device)

            # print("inputs: " + str(inputs))

            # forward + backward + optimize
            # outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - labels.size(): " + str(labels.size()))
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs.size(): " + str(inputs.size()))
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs: " + str(inputs))

            time_start_network_forward = util.timing.date_time_start()
            outputs = self.model(inputs)
            # print("Time used for network forward: " + str(util.timing.milliseconds_since(time_start_network_forward)))

            # print(">>> outputs.size(): " + str(outputs.size()))

            # print(">>> labels.size() : " + str(labels.size()))
            # print("labels: " + str(labels))
            # warp_ctc_loss_interface.
            # print(">>> labels_one_dimensional.size() : " + str(labels_one_dimensional.size()))
            # print("labels_one_dimensional: " + str(labels_one_dimensional))

            # print("outputs: " + str(outputs))
            # print("outputs.size(): " + str(outputs.size()))
            # print("labels: " + str(labels))
            if inputs_is_list:
                number_of_examples = len(inputs)
            else:
                number_of_examples = inputs.size(0)

            time_start_ctc_loss_computation = util.timing.date_time_start()
            # print("trainer - outputs.size(): " + str(outputs.size()))
            loss = self.warp_ctc_loss_interface.compute_ctc_loss(outputs,
                                                                 labels,
                                                                 number_of_examples,
                                                                 self.model_properties.width_reduction_factor)

            # print("Time used for ctc loss computation: " + str(util.timing.milliseconds_since(time_start_ctc_loss_computation)))

            # See: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/train.py
            # The averaging seems to help learning (but a smaller learning rate
            # might have the same effect!)
            loss = loss / number_of_examples  # average the loss by minibatch size

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.item()

            # print("loss: " + str(loss))
            # loss = criterion(outputs, labels)

            time_start_loss_backward = util.timing.date_time_start()

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.model.zero_grad()

            # get_dot = modules.find_bad_gradients.register_hooks(outputs)
            loss.backward()

            # https://discuss.pytorch.org/t/how-to-check-for-vanishing-exploding-gradients/9019/4
            #for p, n in zip(self.model.parameters(), self.model._all_weights[0]):
            #    if n[:6] == 'weight':
            #        print('===========\ngradient:{}\n----------\n{}'.format(n, p.grad))

            # for name, p in self.model.named_parameters():
            #         print('===========\ngradient {} \n----------\n{}'.format(name, p.grad))


            # dot = get_dot()
            # dot.save('mdlstm_ctc_no_data_parallel_find_bad_gradients-clamp-pad-function.dot')
            # render('dot', 'png', 'mdlstm_ctc_mnist_find_bad_gradients.dot')
            # print("Time used for loss backward: " + str(util.timing.milliseconds_since(time_start_loss_backward)))

            # raise RuntimeError("stopping after find bad gradients")

            # Perform step including gradient clipping
            made_gradient_norm_based_correction, total_norm = self.optimizer.step()
            print("trainer - total norm: " + str(total_norm))

            if made_gradient_norm_based_correction:
                num_gradient_corrections += 1
            gradient_norms_sum += total_norm

            # print statistics
            # print("loss.data: " + str(loss.data))
            # print("loss.data[0]: " + str(loss.data[0]))
            running_loss += loss_value
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            # See: https://stackoverflow.com/questions/5598181/python-multiple-prints-on-the-same-line
            # print(str(i)+",", end="", flush=True)
            if i % 10 == 9:  # print every 10 mini-batches
                end = time.time()
                running_time = end - start
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / 10) +
                      " Running time: " + str(running_time))
                average_norm = gradient_norms_sum / 10
                print("Number of gradient norm-based corrections: " + str(num_gradient_corrections))
                print("Average gradient total norm: " + str(average_norm))
                running_loss = 0.0
                num_gradient_corrections = 0
                gradient_norms_sum = 0

                percent = (i + 1) / float(len(train_loader))
                examples_processed = (i + 1) * batch_size
                total_examples = len(train_loader.dataset)
                print("Processed " + str(examples_processed) + " of " + str(total_examples) + " examples in this epoch")
                print(">>> Time used in current epoch: " +
                      str(util.timing.time_since_and_expected_remaining_time(time_start, percent)))
                sys.stdout.flush()

    def drop_checkpoint(self, opt, epoch, valid_stats:ValidationStats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, torch.nn.DataParallel)
                      else self.model)
        # real_generator = (real_model.generator.module
        #                   if isinstance(real_model.generator, torch.nn.DataParallel)
        #                   else real_model.generator)
        #
        model_state_dict = real_model.state_dict()

        # Not sure what the generator is for and if it is really needed
        #model_state_dict = {k: v for k, v in model_state_dict.items()
        #                    if 'generator' not in k}
        #generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            #'generator': generator_state_dict,
            'opt': opt,
            'epoch': epoch,
            'optim': self.optimizer,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(), epoch))
