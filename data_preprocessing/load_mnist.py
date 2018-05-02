import os.path
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import util.image_visualization
import matplotlib.pyplot as plt
import torchvision
from util.image_input_transformer import ImageInputTransformer


def get_train_loader(batch_size):
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 32*32
    trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    #trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    return train_loader


def get_test_loader(batch_size):
    # http://docs.python-guide.org/en/latest/writing/structure/

    # see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
    project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

    # exec(open(project_root + "shared_imports.py").read())

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()

    root = project_root + '/data'
    download = False  # download MNIST dataset or not

    # Scaling to size 32*32
    trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    #trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)
    return test_loader


def get_first_image():

    train_loader = get_train_loader()

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    #util.image_visualization.show_random_batch_from_image_loader_with_class_labels(train_loader, classes)

    images_iteration = iter(train_loader)
    images, labels = images_iteration.next()
    # Show the first image
    image = images[0]
    print("image: " + str(image))
    #util.image_visualization.imshow(torchvision.utils.make_grid(image))
    #plt.show()
    transformed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(image)
    print("transformed_image: " + str(transformed_image))
    util.image_visualization.imshow(torchvision.utils.make_grid(transformed_image))
    plt.show()

    return image


#get_first_image()