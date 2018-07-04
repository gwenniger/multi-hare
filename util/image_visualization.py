import matplotlib.pyplot as plt
import numpy as np

import torchvision


# functions to show an image

def imshow_tensor_2d(img):
    img = img / 2 + 0.5  # unnormalize
    # npimg = img.numpy()
    npimg = img.detach().numpy()
    plt.imshow(npimg)
    plt.show()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



def show_random_batch_from_image_loader_with_class_labels(image_loader, classes):
    # get some random training images
    dataiter = iter(image_loader)
    images, labels = dataiter.next()

    # print labels
    print("Class labels:")
    print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

    # show images
    imshow(torchvision.utils.make_grid(images))
    plt.show()

