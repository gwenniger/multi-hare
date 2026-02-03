import torch
import modules.train_multi_dimensional_rnn_ctc
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
import os

cwd = os.getcwd()
EXPERIMENT_FOLDER = cwd + "/VariableLengthMnistExperiment/"

def main():
    """
    The aim of this test is to check all the main components of the pipeline, that is
    training, validation and testing, while using the ctc loss function during training and
    the ctcdecoder.
    This test uses the variable length mnist data, which is automatically generated from the
    built-in torch MNIST dataset.
    Note that below in the arguments there are some odd looking values "A" --"G"
    These names are chose deliberately as such for debugging purposes. These arguments need to be
    specified but are not actually used for the variable_length_mnist experiment.
    Note that "iam_words_training_test.py" gives more sensible
    (file) names for the case where these arguments are actually used.
    """


    Path(EXPERIMENT_FOLDER).mkdir(parents=True, exist_ok=True)


    argv = ["-examples_database_data_type","variable_length_mnist",
            "-data_permutation_file_path","A",
            "-vocabulary_file_path","B",
#           "-mdlstm_layer_sizes","2", "10", "50",
            "-mdlstm_layer_sizes", "4", "20", "100",
            "-language_model_file_path", "C",
            "-no_language_model",
            "-language_model_weight","0",
            "-word_insertion_penalty","0",
            # Learning rate as used in the paper "No Padding Please: Efficient Neural Handwriting Recognition"
            "-learning_rate", "0.005",
            "-use_leaky_lp_cells",
            "-use_dropout",
            "-use_regular_mdlstm_layers",
            "-load_entire_dataset_beforehand",
            #"-use_on_demand_example_loading",
            #"-use_example_packing",
            "-no_example_packing",
            "-no_bias_in_block_strided_convolution",
            "-save_score_table_file_path",EXPERIMENT_FOLDER + "mnist-ctr-results-table.txt",
            "-use_network_structure_bluche",
            #"-use_unique_weights_for_each_directions_in_fully_connected_layer",
            # Weight sharing across directions in fully connected layer, as used in the paper
            # "No Padding Please: Efficient Neural Handwriting Recognition"
            "-share_weights_across_directions_in_fully_connected_layer",
            "-dataset_save_or_load_file_path","E",
            "-iam_database_line_images_root_folder_path","F",
            "-iam_database_lines_file_path","G",
            "-use_fractions_based_data_split",
            "-gpuid", "0",
            # You should lower the number of epochs if you just want to quickly test all the steps,
            # but note that to properly learn quite some epochs are required.
            "-epochs", "50",
            "-batch_size", "512",
            "-save_model", EXPERIMENT_FOLDER + "model",  # ,
            #"-train_from", "MODEL_PATH"   #Specify your model path here if you want to resume from an earlier checkpoint
            #"-train_from", EXPERIMENT_FOLDER + "model_acc_0.00_cer_92.229_wer_100.000_e5.pt"
            ]

    modules.train_multi_dimensional_rnn_ctc.main(argv)


if __name__ == "__main__":
    main()
