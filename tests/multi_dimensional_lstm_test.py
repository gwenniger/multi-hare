import torch
from modules.multi_dimensional_lstm import MultiDimensionalLSTM
from util.tensor_utils import TensorUtils

class MultiDimensionalLSTMTest:

    def __init__(self, mdlstm):
        self.mdlstm = mdlstm

    @staticmethod
    def create_multi_dimensional_lstm_test():
        layer_index = 0
        input_channels = 1
        hidden_state_size = 1
        compute_multi_directional = False
        clamp_gradients = False
        use_dropout = False
        use_examples_packing = False
        return MultiDimensionalLSTMTest(MultiDimensionalLSTM.\
            create_multi_dimensional_lstm_fast(layer_index, input_channels,
                                               hidden_state_size, compute_multi_directional,
                                               clamp_gradients, use_dropout, use_examples_packing
                                               ).cuda())

    def get_mdlstm_activations_with_and_without_packing(self, input_tensor, input_tensor_list):
        activations = self.mdlstm(input_tensor)

        # print("activations: " + str(activations))

        self.mdlstm.set_use_examples_packing(True)
        activations_with_examples_packing = self.mdlstm(input_tensor_list)
        # print("activations_with_examples_packing: " + str(activations_with_examples_packing))
        return activations, activations_with_examples_packing

    @staticmethod
    def assert_results_are_same_with_and_without_packing(activations_example_without_packing,
                                                         activations_example_with_packing):
        if not TensorUtils.tensors_are_equal(activations_example_without_packing, activations_example_with_packing ):
            raise RuntimeError("Error: expected the same activations for MDLSTM forward computation" +
                               "with or without packing, but got different results:\n - without packing:\n"
                               + str(activations_example_without_packing) + "\n - with packing: \n" +
                               str(activations_example_with_packing))

    @staticmethod
    def test_simple_normal_and_packed_mdlstm_computation_produce_same_results():
        mdlstm_test = MultiDimensionalLSTMTest.create_multi_dimensional_lstm_test()
        input_tensor = torch.ones(1, 1, 2, 2).cuda()
        input_tensor_list = list([torch.ones(1, 2, 2).cuda()])
        activations, activations_with_examples_packing = mdlstm_test.\
            get_mdlstm_activations_with_and_without_packing(input_tensor,
                                                            input_tensor_list)

        print("activations_without_examples_packing: " + str(activations))
        print("activations_with_examples_packing: " + str(activations_with_examples_packing))

        if not TensorUtils.tensors_are_equal(activations, activations_with_examples_packing[0]):
            raise RuntimeError("Error: expected the same activations for MDLSTM forward computation" +
                               "with or without packing, but got different results")

    @staticmethod
    def test_simple_normal_and_packed_mdlstm_computation_multiple_examples_produce_same_results():
        mdlstm_test = MultiDimensionalLSTMTest.create_multi_dimensional_lstm_test()
        input_tensor = torch.ones(2, 1, 2, 2).cuda()
        input_tensor_list = list([torch.ones(1, 2, 2).cuda(), torch.ones(1, 2, 2).cuda()])
        activations_without_example_packing, activations_with_examples_packing = mdlstm_test. \
            get_mdlstm_activations_with_and_without_packing(input_tensor,
                                                            input_tensor_list)

        activations_example_without_packing = activations_without_example_packing[0]
        print("activations_without_examples_packing[0]:" + str(activations_without_example_packing[0]))
        activations_example_with_packing = activations_with_examples_packing[0]
        print("activations_with_examples_packing[0]:" + str(activations_with_examples_packing[0]))
        MultiDimensionalLSTMTest.assert_results_are_same_with_and_without_packing(activations_example_without_packing,
                                                                                  activations_example_with_packing)

        activations_example_without_packing = activations_without_example_packing[1]
        print("activations_without_examples_packing1]:" + str(activations_without_example_packing[1]))
        activations_example_with_packing = activations_with_examples_packing[1]
        print("activations_with_examples_packing[1]:" + str(activations_with_examples_packing[1]))
        MultiDimensionalLSTMTest.assert_results_are_same_with_and_without_packing(activations_example_without_packing,
                                                                                  activations_example_with_packing)

    """
    A more complicated test scenario. In the examples packing example, there 
    are two short example tensors and one longer. If we compute the activations
    for the shorter examples without packing, the activations should still be 
    the same as when computed when incorporated in the longer examples list which 
    is processed using packing. This is tested here.
    """
    @staticmethod
    def test_normal_and_packing_incorporated_mdlstm_computation_produce_same_results_two():
        mdlstm_test = MultiDimensionalLSTMTest.create_multi_dimensional_lstm_test()
        # input_tensor = torch.ones(2, 1, 2, 2).cuda()
        # input_tensor_list = list([torch.ones(1, 2, 8).cuda(),
        #                           torch.ones(1, 2, 2).cuda(),
        #                          torch.ones(1, 2, 2).cuda()])
        input_tensor = torch.ones(2, 1, 1, 2).cuda()
        input_tensor_list = list([torch.ones(1, 1, 8).cuda(),
                                  torch.ones(1, 1, 2).cuda(),
                                  torch.ones(1, 1, 2).cuda()])

        activations, activations_with_examples_packing = mdlstm_test. \
            get_mdlstm_activations_with_and_without_packing(input_tensor,
                                                            input_tensor_list)

        # print("activations_with_examples_packing: " + str(activations_with_examples_packing))

        activations_example_without_packing = activations[0, :, :, :]
        activations_example_with_packing = activations_with_examples_packing[1]
        MultiDimensionalLSTMTest.assert_results_are_same_with_and_without_packing(activations_example_without_packing,
                                                                                  activations_example_with_packing)

        activations_example_without_packing = activations[1, :, :, :]
        activations_example_with_packing = activations_with_examples_packing[2]
        MultiDimensionalLSTMTest.assert_results_are_same_with_and_without_packing(activations_example_without_packing,
                                                                                  activations_example_with_packing)


def main():
    MultiDimensionalLSTMTest.test_simple_normal_and_packed_mdlstm_computation_produce_same_results()
    MultiDimensionalLSTMTest.test_simple_normal_and_packed_mdlstm_computation_multiple_examples_produce_same_results()
    MultiDimensionalLSTMTest.test_normal_and_packing_incorporated_mdlstm_computation_produce_same_results_two()


if __name__ == "__main__":
    main()
