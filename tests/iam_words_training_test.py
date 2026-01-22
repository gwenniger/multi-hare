import sys
import modules.train_multi_dimensional_rnn_ctc
import torch
#torch.multiprocessing.set_sharing_strategy('file_system')



def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Usage: >>> iam_word_training_test IAM_DATA_ROOT_FOLDER EXPERIMENT_ROOT_FOLDER")
    iam_data_root_folder = sys.argv[1]
    experiment_folder = sys.argv[2]

    argv = ["-examples_database_data_type","iam_words",
            "-data_permutation_file_path",experiment_folder + "data_permutation.txt",
            "-vocabulary_file_path", experiment_folder + "vocabulary.txt",
#           "-mdlstm_layer_sizes","2", "10", "50",
            "-mdlstm_layer_sizes", "4", "20", "100",
            "-language_model_file_path", experiment_folder + "language_model",
            "-no_language_model",
            "-language_model_weight","0",
            "-word_insertion_penalty","0",
            # Learning rate as used in the paper "No Padding Please: Efficient Neural Handwriting Recognition"
            "-learning_rate", "0.005",
            "-use_leaky_lp_cells",
            "-use_dropout",
            "-use_four_pixel_input_blocks",
            "-use_regular_mdlstm_layers",
            "-load_entire_dataset_beforehand",
            #"-use_on_demand_example_loading",
            "-use_example_packing",
            #"-no_example_packing",
            "-no_bias_in_block_strided_convolution",
            "-save_score_table_file_path",experiment_folder + "iam-words-training-results-table.txt",
            "-use_network_structure_bluche",
            #"-use_unique_weights_for_each_directions_in_fully_connected_layer",
            # Weight sharing across directions in fully connected layer, as used in the paper
            # "No Padding Please: Efficient Neural Handwriting Recognition"
            "-share_weights_across_directions_in_fully_connected_layer",
            "-dataset_save_or_load_file_path", experiment_folder + "words_dataset_prepared",
            "-iam_database_line_images_root_folder_path", iam_data_root_folder + "words/",
            "-iam_database_lines_file_path", iam_data_root_folder + "ascii/words.txt",
            "-use_fractions_based_data_split",
            "-gpuid", "0",
            "-epochs", "50",
            # Estimated possible batch size, based on single 8G GPU. Was tested to work in
            # "No padding please: Efficient Neural Handwriting Recognition" paper to work
            # with batch size 200 using two 11178 MB GPUs, noting the results get merged
            # on a single one of these GPUs so memory requirements remain the same with
            # either one or two GPUs (though speed of course increases with two GPUs)
            "-batch_size", "128",
            "-save_model", experiment_folder +"model"
            ]

    modules.train_multi_dimensional_rnn_ctc.main(argv)


if __name__ == "__main__":
    main()
