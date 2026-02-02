# Run some basic tests after installation, to check if everything is working OK
export PYTHONPATH=.
echo "TEST 1: training on the variable-length mnist dataset, using ctc loss"
## This test requires no data to be downloaded and nothing extra
python tests/train_multi_dimensional_rnn_ctc_test.py 


echo "TEST 2: training on eht IAM words dataset, using ctc loss"
# This test requires the IAM words dataset 
IAM_DATA_ROOT_FOLDER=""    # Change this to the actual path to the IAM data folder
EXPERIMENT_ROOT_FOLDER="" # Change to this the path of the folder where you want to experiment output to be written to 
python tests/iam_words_training_test.py ${IAM_DATA_ROOT_FOLDER} ${EXPERIMENT_ROOT_FOLDER}

