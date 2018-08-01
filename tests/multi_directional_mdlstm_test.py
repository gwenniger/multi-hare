import torch
from modules.multi_dimensional_lstm import MultiDimensionalLSTM
import util.image_visualization
from modules.mdlstm_examples_packing import MDLSTMExamplesPacking
from util.tensor_utils import TensorUtils

"""
This test class tests multi-directional MDLSTM by comparing its output by the output 
produced by four one-directional MDLSTMs that use the same weights, which are copied 
over from the 4-directional MDLSTM. This type of indirect testing is arguably the best 
we can do, since it is very hard to test directly whether the 4-directional MDLSTM 
"computes the right thing". There are many things that could go wrong with 
the 4-directional MDLSTM computation, but the 1-directional MDLSTM computation is 
simpler and has been shown to work already. Therefore, if an 
appropriately flipped  input (for the given direction) produces an output for the
one-directional MDLSTM of the direction, that is equal (after flipping back)
to the output of the 4-directional MDLSTM for that direction;
this gives quite some confidence that the 4-directional MDLSTM implementation is correct, 
assuming the 1-directional MDLSTM implementation was correct.
"""


class MultiDirectionalMDLSTMTest:

    def __init__(self, multi_directional_mdlstm):
        self.multi_directional_mdlstm = multi_directional_mdlstm

    @staticmethod
    def create_multi_directional_mdlstm_test():
        input_channels = 1
        hidden_states_size = 2

        multi_directional_mdlstm = MultiDimensionalLSTM.create_multi_dimensional_lstm_fully_parallel(
            layer_index=0, input_channels=input_channels, hidden_states_size=hidden_states_size,
            compute_multi_directional=True, clamp_gradients=False, use_dropout=False,
            use_example_packing=True, nonlinearity="tanh")

        return MultiDirectionalMDLSTMTest(multi_directional_mdlstm)

    @staticmethod
    def create_test_tensor_simplest():
        test_tensor = torch.ones(1, 1, 2).cuda()
        test_tensor[:, 0, 1] = 0

        test_tensor_2d = test_tensor.squeeze(0)
        print("Visualizing test tensor...")
        util.image_visualization.imshow_tensor_2d(test_tensor_2d.cpu())
        return test_tensor

    @staticmethod
    def create_test_tensor_very_simple():
        test_tensor = torch.ones(1, 2, 2).cuda()
        test_tensor[:, 0, 0] = 0
        test_tensor[:, 0, 1] = 0
        test_tensor[:, 1, 1] = 0

        test_tensor_2d = test_tensor.squeeze(0)
        print("Visualizing test tensor...")
        util.image_visualization.imshow_tensor_2d(test_tensor_2d.cpu())
        return test_tensor


    @staticmethod
    def create_test_tensor_simple():
        test_tensor = torch.ones(1, 2, 4).cuda()
        test_tensor[:, 0, 0] = 0
        test_tensor[:, 0, 1] = 0
        test_tensor[:, 1, 1] = 0

        test_tensor_2d = test_tensor.squeeze(0)
        print("Visualizing test tensor...")
        util.image_visualization.imshow_tensor_2d(test_tensor_2d.cpu())
        return test_tensor

    @staticmethod
    def create_test_tensor_intermediate():
        test_tensor = torch.ones(1, 4, 8).cuda()
        test_tensor[:, 1, 1:3] = 0
        test_tensor[:, 1, 5] = 0
        test_tensor[:, 3, 7:8] = 0

        test_tensor_2d = test_tensor.squeeze(0)
        print("Visualizing test tensor...")
        util.image_visualization.imshow_tensor_2d(test_tensor_2d.cpu())
        return test_tensor

    @staticmethod
    def create_test_tensor_complex():
        test_tensor = torch.ones(1, 8, 16).cuda()
        test_tensor[:, 1, 1:6] = 0
        test_tensor[:, 2, 1:6] = 0
        test_tensor[:, 3, 1:6] = 0
        test_tensor[:, 1, 10:12] = 0
        test_tensor[:, 2, 11:13] = 0
        test_tensor[:, 3, 12:14] = 0
        test_tensor[:, 4, 13:15] = 0

        test_tensor_2d = test_tensor.squeeze(0)
        print("Visualizing test tensor...")
        util.image_visualization.imshow_tensor_2d(test_tensor_2d.cpu())
        return test_tensor

    @staticmethod
    def assert_input_convolution_weights_are_equal(multi_directional_mdlstm, one_directional_mdlstm,
                                                   direction_index: int):
        multi_directional_mdlstm_input_convolution_computation =\
            multi_directional_mdlstm.mdlstm_parameters.\
            parallel_multiple_input_convolutions_computation

        out_channels_size = multi_directional_mdlstm_input_convolution_computation.\
            parallel_convolution.weight.size(0)
        out_channels_per_direction = out_channels_size / 4
        start_index = int(out_channels_per_direction * direction_index)
        end_index = int(out_channels_per_direction * (direction_index + 1))

        one_directional_weight_from_multi_directional =\
            multi_directional_mdlstm_input_convolution_computation.parallel_convolution.weight[start_index:end_index,
                                                                                               :, :, :]

        one_directional_bias_from_multi_directional =\
            multi_directional_mdlstm_input_convolution_computation.parallel_convolution.bias[start_index:end_index]

        one_directional_mdlstm_input_convolution_computation =\
            one_directional_mdlstm.mdlstm_parameters.parallel_multiple_input_convolutions_computation

        if not TensorUtils.tensors_are_equal(
                one_directional_weight_from_multi_directional,
                one_directional_mdlstm_input_convolution_computation.parallel_convolution.weight):
                raise RuntimeError("Error: the weight matrices for the input convolution computation for " +
                                   "multi-directional MDLSTM and the corresponding one-directional MDLSTM" +
                                   "are not the same")
        if not TensorUtils.tensors_are_equal(
                one_directional_bias_from_multi_directional,
                one_directional_mdlstm_input_convolution_computation.parallel_convolution.bias):
            raise RuntimeError("Error: the bias matrices for the input convolution computation for " +
                               "multi-directional MDLSTM and the corresponding one-directional MDLSTM" +
                               "are not the same")

    @staticmethod
    def assert_output_gate_memory_state_convolution_weights_are_equal(multi_directional_mdlstm, one_directional_mdlstm,
                                                                      direction_index: int):
        multi_directional_mdlstm_output_gate_memory_state_convolution =\
            multi_directional_mdlstm.mdlstm_parameters. \
            output_gate_memory_state_convolution

        out_channels_size = multi_directional_mdlstm_output_gate_memory_state_convolution.weight.size(0)
        out_channels_per_direction = out_channels_size / 4
        start_index = int(out_channels_per_direction * direction_index)
        end_index = int(out_channels_per_direction * (direction_index + 1))

        multi_directional_mdlstm_output_gate_memory_state_convolution_weight_for_direction =\
            multi_directional_mdlstm_output_gate_memory_state_convolution.weight[start_index:end_index, :, :]

        multi_directional_mdlstm_output_gate_memory_state_convolution_bias_for_direction = \
            multi_directional_mdlstm_output_gate_memory_state_convolution.bias[start_index:end_index]

        one_directional_mdlstm_output_gate_memory_state_convolution =\
            one_directional_mdlstm.mdlstm_parameters.output_gate_memory_state_convolution

        if not TensorUtils.tensors_are_equal(
                multi_directional_mdlstm_output_gate_memory_state_convolution_weight_for_direction,
                one_directional_mdlstm_output_gate_memory_state_convolution.weight):
                raise RuntimeError("Error: the weight matrices for the output gate memory state convolution for " +
                                   "multi-directional MDLSTM and the corresponding one-directional MDLSTM" +
                                   "are not the same")
        if not TensorUtils.tensors_are_equal(
                multi_directional_mdlstm_output_gate_memory_state_convolution_bias_for_direction,
                one_directional_mdlstm_output_gate_memory_state_convolution.bias):
            raise RuntimeError("Error: the bias matrices for the output gate memory state convolution for " +
                               "multi-directional MDLSTM and the corresponding one-directional MDLSTM" +
                               "are not the same")

    @staticmethod
    def assert_parallel_hidden_and_memory_state_column_computation_weights_are_equal(
            multi_directional_mdlstm, one_directional_mdlstm, direction_index: int):
        multi_directional_mdlstm_parallel_hidden_and_memory_state_column_computation = \
            multi_directional_mdlstm.mdlstm_parameters. \
                parallel_hidden_and_memory_state_column_computation

        out_channels_size = multi_directional_mdlstm_parallel_hidden_and_memory_state_column_computation.\
            parallel_convolution.weight.size(0)
        out_channels_per_direction = out_channels_size / 4
        start_index = int(out_channels_per_direction * direction_index)
        end_index = int(out_channels_per_direction * (direction_index + 1))

        multi_directional_mdlstm_weight_for_direction = \
            multi_directional_mdlstm_parallel_hidden_and_memory_state_column_computation.\
            parallel_convolution.weight[start_index:end_index, :, :]

        multi_directional_mdlstm_bias_for_direction = \
            multi_directional_mdlstm_parallel_hidden_and_memory_state_column_computation.\
            parallel_convolution.bias[start_index:end_index]

        one_directional_mdlstm_parallel_hidden_and_memory_state_column_computation = \
            one_directional_mdlstm.mdlstm_parameters.parallel_hidden_and_memory_state_column_computation

        if not TensorUtils.tensors_are_equal(
                multi_directional_mdlstm_weight_for_direction,
                one_directional_mdlstm_parallel_hidden_and_memory_state_column_computation.
                parallel_convolution.weight):
            raise RuntimeError("Error: the weight matrices for the" +
                               "one_directional_mdlstm_parallel_hidden_and_memory_state_column_computation" +
                               "multi-directional MDLSTM and the corresponding one-directional MDLSTM" +
                               "are not the same")
        if not TensorUtils.tensors_are_equal(
                multi_directional_mdlstm_bias_for_direction,
                one_directional_mdlstm_parallel_hidden_and_memory_state_column_computation.
                parallel_convolution.bias):
            raise RuntimeError("Error: the bias matrices for the" +
                               "one_directional_mdlstm_parallel_hidden_and_memory_state_column_computation" +
                               "multi-directional MDLSTM and the corresponding one-directional MDLSTM" +
                               "are not the same")

    @staticmethod
    def test_multi_directional_mdlstm_produces_same_results_as_extracted_one_directional_mdlstms(test_tensor):
        multi_directional_mdlstm_test = MultiDirectionalMDLSTMTest.create_multi_directional_mdlstm_test()
        multi_directional_mdlstm_test.multi_directional_mdlstm = multi_directional_mdlstm_test.\
            multi_directional_mdlstm.cuda()
        one_directional_mdlstms = multi_directional_mdlstm_test.\
            multi_directional_mdlstm.create_one_directional_mdlstms_from_multi_directional_mdlstm()

        activations_multi_directional_mdlstm = multi_directional_mdlstm_test.\
            multi_directional_mdlstm(list([test_tensor]))
        print("activations_multi_directional_mdlstm: " + str(activations_multi_directional_mdlstm))
        assert len(activations_multi_directional_mdlstm) == 1

        activations_for_tensor = activations_multi_directional_mdlstm[0]
        print("activations_for_tensor.size(): " + str(activations_for_tensor.size()))
        # if not activations_for_tensor.size(0) == 4:
        #     raise RuntimeError("Error: expected the output tensor to have a size of 4" +
        #                        "for its first dimension, i.e. for 4-directional MDLSTM")

        tensor_flipping_list = MDLSTMExamplesPacking.create_four_directions_tensor_flippings()

        for direction_index, tensor_flipping in enumerate(tensor_flipping_list):
            print(">>> direction_index: " + str(direction_index))
            one_directional_mdlstm = one_directional_mdlstms[direction_index].cuda()
            MultiDirectionalMDLSTMTest.assert_input_convolution_weights_are_equal(
                multi_directional_mdlstm_test.multi_directional_mdlstm, one_directional_mdlstm, direction_index)
            MultiDirectionalMDLSTMTest.assert_output_gate_memory_state_convolution_weights_are_equal(
                multi_directional_mdlstm_test.multi_directional_mdlstm, one_directional_mdlstm, direction_index)
            MultiDirectionalMDLSTMTest.assert_parallel_hidden_and_memory_state_column_computation_weights_are_equal(
                multi_directional_mdlstm_test.multi_directional_mdlstm, one_directional_mdlstm, direction_index)

            test_tensor_flipped = tensor_flipping.flip(test_tensor).cuda()

            activations_one_directional_mdlstm_flipped = one_directional_mdlstm(list([test_tensor_flipped]))
            # Flip activations back to original orientation
            activations_one_directional_mdlstm = \
                tensor_flipping.flip(activations_one_directional_mdlstm_flipped[0])
            start_index = int(direction_index * (activations_for_tensor.size(1) / 4))
            end_index = int((direction_index + 1) * (activations_for_tensor.size(1) / 4))
            print("start_index: " + str(start_index))
            print("end_index: " + str(end_index))

            # Activations are concatenated along the channel dimension
            activations_one_directional_mdlstm_from_four_directional_mdlstm = \
                activations_for_tensor[:, start_index:end_index, :, :]

            print("activations_one_directional_mdlstm_from_four_directional_mdlstm: " +
                  str(activations_one_directional_mdlstm_from_four_directional_mdlstm))
            print("activations_one_directional_mdlstm: " +
                  str(activations_one_directional_mdlstm))
            if not TensorUtils.tensors_are_equal(activations_one_directional_mdlstm,
                                                 activations_one_directional_mdlstm_from_four_directional_mdlstm):
                raise RuntimeError("Error: expected the activation tensors for the one-directional MDLSTM: \n" +
                                   str(activations_one_directional_mdlstm) + "\n and the corresponding ones " +
                                   " of the 4-directional MDLSTM \n" +
                                   str(activations_one_directional_mdlstm_from_four_directional_mdlstm) +
                                   " to be the same.")

    @staticmethod
    def test_simplest_tensor():
        test_tensor = MultiDirectionalMDLSTMTest.create_test_tensor_simplest()
        MultiDirectionalMDLSTMTest.\
            test_multi_directional_mdlstm_produces_same_results_as_extracted_one_directional_mdlstms(test_tensor)

    @staticmethod
    def test_very_simple_tensor():
        test_tensor = MultiDirectionalMDLSTMTest.create_test_tensor_very_simple()
        MultiDirectionalMDLSTMTest. \
            test_multi_directional_mdlstm_produces_same_results_as_extracted_one_directional_mdlstms(test_tensor)

    @staticmethod
    def test_simple_tensor():
        test_tensor = MultiDirectionalMDLSTMTest.create_test_tensor_simple()
        MultiDirectionalMDLSTMTest. \
            test_multi_directional_mdlstm_produces_same_results_as_extracted_one_directional_mdlstms(test_tensor)

    @staticmethod
    def test_intermediate_tensor():
        test_tensor = MultiDirectionalMDLSTMTest.create_test_tensor_intermediate()
        MultiDirectionalMDLSTMTest. \
            test_multi_directional_mdlstm_produces_same_results_as_extracted_one_directional_mdlstms(test_tensor)

    @staticmethod
    def test_complex_tensor():
        test_tensor = MultiDirectionalMDLSTMTest.create_test_tensor_complex()
        MultiDirectionalMDLSTMTest. \
            test_multi_directional_mdlstm_produces_same_results_as_extracted_one_directional_mdlstms(test_tensor)


def main():
    # Test that the 4-directional MDLSTM and corresponding 1-directional MDLSTMs
    # produce the same outputs, for increasingly complex input tensors
    MultiDirectionalMDLSTMTest.test_simplest_tensor()
    MultiDirectionalMDLSTMTest.test_very_simple_tensor()
    MultiDirectionalMDLSTMTest.test_simple_tensor()
    MultiDirectionalMDLSTMTest.test_intermediate_tensor()
    MultiDirectionalMDLSTMTest.test_complex_tensor()


if __name__ == "__main__":
    main()


