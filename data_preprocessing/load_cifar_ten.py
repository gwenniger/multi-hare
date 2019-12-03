import torchvision.transforms as transforms
import torch
import torchvision

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


def get_train_set():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    return trainset


def get_test_set():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    return testset


def get_train_loader(batch_size: int):
    train_loader = torch.utils.data.DataLoader(get_train_set(), batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    return train_loader


def get_test_loader(batch_size: int):
    test_loader = torch.utils.data.DataLoader(get_test_set(), batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    return test_loader
