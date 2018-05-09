import os.path
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import util.image_visualization
import matplotlib.pyplot as plt
import torchvision
from util.image_input_transformer import ImageInputTransformer
from random import randint


def get_train_set():
    transform = transforms.Compose(
        [transforms.Resize((16, 16)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    return trainset


def get_test_set():
    transform = transforms.Compose(
        [transforms.Resize((16, 16)), transforms.ToTensor(),
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
