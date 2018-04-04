import os.path
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import util.image_visualization
import matplotlib.pyplot as plt
import torchvision
from util.image_input_transformer import ImageInputTransformer

# http://docs.python-guide.org/en/latest/writing/structure/

# see: https://stackoverflow.com/questions/2668909/how-to-find-the-real-user-home-directory-using-python
project_root = os.path.expanduser('~') + "/AI/handwriting-recognition/"

#exec(open(project_root + "shared_imports.py").read())


## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
download = False  # download MNIST dataset or not

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

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
