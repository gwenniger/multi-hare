import torch
import modules.train_multi_dimensional_rnn_ctc
torch.multiprocessing.set_sharing_strategy('file_system')



def main():
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
            #"-use_four_pixel_input_blocks",
            "-use_resolution_halving",
            "-use_regular_mdlstm_layers",
            "-load_entire_dataset_beforehand",
            #"-use_on_demand_example_loading",
            #"-use_example_packing",
            "-no_example_packing",
            "-no_bias_in_block_strided_convolution",
            "-save_score_table_file_path","mnist-ctr-results-table.txt",
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
            "-epochs", "50",
            "-batch_size", "256",
            #"-train_from", "MODEL_PATH"   #Specify your model path here if you want to resume from an earlier checkpoint
            ]

    modules.train_multi_dimensional_rnn_ctc.main(argv)


if __name__ == "__main__":
    main()
