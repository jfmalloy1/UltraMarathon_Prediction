### Going through pytorch tutorials
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


def intro_to_tensors():
    """ Intro to tensors from "Tensors" tutorial in pytorch
    """
    ### Generate tensors
    shape = (
        5,
        7,
    )
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    # print(f"Random Tensor: \n {rand_tensor} \n")
    # print(f"Ones Tensor: \n {ones_tensor} \n")
    # print(f"Zeros Tensor: \n {zeros_tensor}")

    ### Tensor attributes
    tensor = torch.rand((
        3,
        4,
    ))
    print(f"Tensor: \n {tensor} \n")

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    print(tensor, "\n")
    tensor.add_(5)
    print(f"Tensor + 5: \n {tensor} \n")


def download_FashionMNIST():
    training_data = datasets.FashionMNIST(root="data",
                                          train=True,
                                          download=True,
                                          transform=ToTensor())

    test_data = datasets.FashionMNIST(root="data",
                                      train=False,
                                      download=True,
                                      transform=ToTensor())

    return training_data, test_data


def main():
    intro_to_tensors()

    training_data, test_data = download_FashionMNIST()

    #Start at "Creating a Custom Dataset in https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


if __name__ == "__main__":
    main()
