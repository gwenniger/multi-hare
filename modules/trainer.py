import torch
import torch.nn
import time
from util.utils import Utils
from modules.size_two_dimensional import SizeTwoDimensional
import util.timing
import util.tensor_utils
from data_preprocessing.iam_database_preprocessing.iam_lines_dataset import IamLinesDataset
import sys
from modules.gradient_clipping import GradientClipping
import torch.nn


class ModelProperties:

    def __init__(self, image_input_is_unsigned_int, width_reduction_factor: int):
        self.image_input_is_unsigned_int = image_input_is_unsigned_int
        self.width_reduction_factor = width_reduction_factor


class Trainer:

    def __init__(self, model, optimizer,
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
    def check_inputs_is_right_type(inputs):
        if Utils.use_cuda():
            expected_type_instance = torch.cuda.ByteTensor()
        else:
            expected_type_instance = torch.ByteTensor()

        if inputs.type() != expected_type_instance.type():
            raise RuntimeError("Error: expected a " + str(expected_type_instance.type()) + " type image tensor" +
                               " but got : " + str(inputs.type()))

    def train_one_epoch(self, train_loader, epoch: int, start: int, batch_size,
                        device, report_func=None):
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

            # The format expected by warp_ctc, which reserves the 0 label for blanks
            number_of_zeros = util.tensor_utils.TensorUtils.number_of_zeros(labels)
            # Check that labels indeed start from 1 as required
            if number_of_zeros != 0:
                raise RuntimeError("Error: labels tensor contains zeros, which is " +
                                   " not allowed, since 0 is reserved for blanks")

            if Utils.use_cuda():
                inputs = inputs.to(device)

            # If the image input comes in the form of unsigned ints, they need to
            # be converted to floats (after moving to GPU, i.e. directly on GPU
            # which is faster)
            if self.model_properties.image_input_is_unsigned_int:
                Trainer.check_inputs_is_right_type(inputs)
                inputs = IamLinesDataset.convert_unsigned_int_image_tensor_to_float_image_tensor(inputs)

            # Set requires_grad(True) directly and only for the input
            inputs.requires_grad_(True)

            # wrap them in Variable
            # labels = Variable(labels)  # Labels need no gradient apparently
            # if Utils.use_cuda():

            # Labels must remain on CPU for warp-ctc loss
            # labels = labels.to(device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # print("inputs: " + str(inputs))

            # forward + backward + optimize
            # outputs = multi_dimensional_rnn(Variable(inputs))  # For "Net" (Le Net)
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - labels.size(): " + str(labels.size()))
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs.size(): " + str(inputs.size()))
            # print("train_multi_dimensional_rnn_ctc.train_mdrnn - inputs: " + str(inputs))

            time_start_network_forward = time.time()
            outputs = self.model(inputs)
            # print("Time used for network forward: " + str(util.timing.time_since(time_start_network_forward)))

            # print(">>> outputs.size(): " + str(outputs.size()))

            # print(">>> labels.size() : " + str(labels.size()))
            # print("labels: " + str(labels))
            # warp_ctc_loss_interface.
            # print(">>> labels_one_dimensional.size() : " + str(labels_one_dimensional.size()))
            # print("labels_one_dimensional: " + str(labels_one_dimensional))

            # print("outputs: " + str(outputs))
            # print("outputs.size(): " + str(outputs.size()))
            # print("labels: " + str(labels))
            number_of_examples = inputs.size(0)

            time_start_ctc_loss_computation = time.time()
            loss = self.warp_ctc_loss_interface.compute_ctc_loss(outputs,
                                                                 labels,
                                                                 number_of_examples,
                                                                 self.model_properties.width_reduction_factor)

            # print("Time used for ctc loss computation: " + str(util.timing.time_since(time_start_ctc_loss_computation)))

            # See: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/train.py
            # The averaging seems to help learning (but a smaller learning rate
            # might have the same effect!)
            loss = loss / inputs.size(0)  # average the loss by minibatch size

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.item()

            # print("loss: " + str(loss))
            # loss = criterion(outputs, labels)

            time_start_loss_backward = time.time()
            loss.backward()
            # print("Time used for loss backward: " + str(util.timing.time_since(time_start_loss_backward)))

            # Perform gradient clipping
            made_gradient_norm_based_correction, total_norm = \
                GradientClipping.clip_gradient_norm(self.model)
            if made_gradient_norm_based_correction:
                num_gradient_corrections += 1
            gradient_norms_sum += total_norm

            # if not (loss_sum == inf or loss_sum == -inf):
            self.optimizer.step()

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
                      (epoch + 1, i + 1, running_loss / 10) +
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

    def drop_checkpoint(self, opt, epoch, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, torch.nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, torch.nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(), epoch))
