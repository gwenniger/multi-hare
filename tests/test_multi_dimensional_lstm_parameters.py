import torch
import modules.multi_dimensional_lstm_parameters


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

        mdlstm_parameters = modules.multi_dimensional_lstm_parameters.\
            MultiDirectionalMultiDimensionalLSTMParametersFullyParallel.\
                create_multi_directional_multi_dimensional_leaky_lp_cell_parameters_fully_parallel(
                    TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE,
                    TestMultiDimensionalLSTMParameters.INPUT_CHANNELS,
                    TestMultiDimensionalLSTMParameters.CLAMP_GRADIENTS,
                    TestMultiDimensionalLSTMParameters.USE_DROPOUT,
                    TestMultiDimensionalLSTMParameters.NUMBER_OF_DIRECTIONS)
        mask_width = TestMultiDimensionalLSTMParameters.NUMBER_OF_COLUMNS + \
            TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT - 1
        mask = torch.ones(1, 2800,
                          TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT)
        skewed_images_variable = torch.ones(1,
                                            TestMultiDimensionalLSTMParameters.INPUT_CHANNELS *
                                            TestMultiDimensionalLSTMParameters.NUMBER_OF_DIRECTIONS,
                                            TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT,
                                            mask_width)
        while True:
            mdlstm_parameters.reset_next_input_column_index()
            for i in range(0, TestMultiDimensionalLSTMParameters.NUMBER_OF_COLUMNS):
                previous_hidden_state_column = torch.zeros(
                    1, TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE * 4,
                    TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT)
                previous_memory_state_column = torch.zeros(
                    1, TestMultiDimensionalLSTMParameters.HIDDEN_STATES_SIZE * 4,
                    TestMultiDimensionalLSTMParameters.IMAGE_HEIGHT)
                # This function call does not yet seem to cause a memory leak
                mdlstm_parameters.prepare_input_convolutions(skewed_images_variable)

                # This function call seems to be causing the memory leak
                mdlstm_parameters.prepare_computation_next_column_functions(
                    previous_hidden_state_column,
                    previous_memory_state_column,
                    mask)


def main():
    print("Testing...")
    TestMultiDimensionalLSTMParameters.test_mdlstm_parameters_not_leaking_memory()


if __name__ == "__main__":
    main()

