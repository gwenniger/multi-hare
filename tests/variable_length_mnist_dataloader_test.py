import torch
import modules.train_multi_dimensional_rnn_ctc
# Setting the sharing strategy to "file_system" is necessary to make the test succeed
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    batch_size = 256
    minimize_horizontal_padding = True
    train_loader, test_loader = modules.train_multi_dimensional_rnn_ctc.get_variable_length_mnist_dataloaders(
        batch_size, minimize_horizontal_padding)
    for i, data in enumerate(train_loader, 0):
        continue

    for i, data in enumerate(test_loader, 0):
        continue


if __name__ == "__main__":
    main()
