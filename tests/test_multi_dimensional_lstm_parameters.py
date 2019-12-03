import torch
import modules.multi_dimensional_lstm_parameters
import torch.nn as nn

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class TestMultiDimensionalLSTMParameters:
    HIDDEN_STATES_SIZE = 50
    INPUT_CHANNELS = 20
    USE_DROPOUT = False
    CLAMP_GRADIENTS = False
    USE_LEAKY_LP_CELLST = True
    NUMBER_OF_COLUMNS = 100
    IMAGE_HEIGHT = 128
    NUMBER_OF_DIRECTIONS = 4
    NUMBER_OF_PAIRED_INPUT_WEIGHTINGS = 7 * NUMBER_OF_DIRECTIONS

    def __int__(self):
        return

    @staticmethod
    def test_mdlstm_parameters_not_leaking_memory():
        """
        This test strives to test if mdlstm parameters are leaking memory, by observing
        memory usage over time with top while running this test. However, looking at
        virtual and reserved memory with top, it is not very clear whether memory is actually leaked.
        What is clear is that the involved methods cause memory usage to fluctuate quite a bit.

        Update 13-6-2019: Adapted the code to use the GPU

        :return:
        """

        # http://pytorch.org/docs/master/notes/cuda.html
        device = torch.device("cuda:0")

        mdlstm_parameters = modules.multi_dimensional_lstm_parameters.\
            MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
                create_multi_directional_multi_dimensional_leaky_lp_cell_parameters_fully_parallel(
                    TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE,
                    TestMultiDimensionalLSTMParameters.INPUT_CHANNELS,
                    TestMultiDimensionalLSTMParameters.CLAMP_GRADIENTS,
                    TestMultiDimensionalLSTMParameters.USE_DROPOUT,
                    TestMultiDimensionalLSTMParameters.NUMBER_OF_DIRECTIONS)
        mdlstm_parameters =  mdlstm_parameters.to(device)
        mask_width = TestMultiDimensionalLSTMParameters.NUMBER_OF_COLUMNS + \
            TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT - 1
        mask = torch.ones(1, 2800,
                          TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT)
        mask = mask.to(device)
        skewed_images_variable = torch.ones(1,
                                            TestMultiDimensionalLSTMParameters.INPUT_CHANNELS *
                                            TestMultiDimensionalLSTMParameters.NUMBER_OF_DIRECTIONS,
                                            TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT,
                                            mask_width)
        skewed_images_variable = skewed_images_variable.to(device)
        while True:
            mdlstm_parameters.reset_next_input_column_index()
            for i in range(0, TestMultiDimensionalLSTMParameters.NUMBER_OF_COLUMNS):
                previous_hidden_state_column = torch.zeros(
                    1, TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE * 4,
                    TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT)
                previous_hidden_state_column = previous_hidden_state_column.to(device)
                previous_memory_state_column = torch.zeros(
                    1, TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE * 4,
                    TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT)
                previous_memory_state_column = previous_memory_state_column.to(device)
                # This function call does not yet seem to cause a memory leak
                mdlstm_parameters.prepare_input_convolutions(skewed_images_variable)

                # This function call seems to be causing the memory leak
                mdlstm_parameters.prepare_computation_next_column_functions(
                    previous_hidden_state_column,
                    previous_memory_state_column,
                    mask)

    @staticmethod
    def test_conv1d_with_grouping_itself_leaks_memory():
        """
        This test tries to establish whether a conv1d with grouping by itself causes memory loss.
        However, this does not seem to be the case.
        :return:
        """
        groups = 56
        input_states_size = TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE * groups
        output_states_size = TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE * groups
        parallel_convolution = nn.Conv1d(input_states_size, output_states_size, 1,
                                         groups=groups)

        while True:
            input_tensor = torch.zeros(
                    1, input_states_size,
                    TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT)
            parallel_convolution.forward(input_tensor)


def main():
    print("Testing...")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    TestMultiDimensionalLSTMParameters.test_mdlstm_parameters_not_leaking_memory()
    # TestMultiDimensionalLSTMParameters.test_conv1d_with_grouping_itself_leaks_memory()


if __name__ == "__main__":
    main()

