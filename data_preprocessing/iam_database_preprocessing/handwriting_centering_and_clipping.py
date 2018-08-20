from skimage import io, transform
import numpy as np
import torch
import util.image_visualization
from data_preprocessing.iam_database_preprocessing.iam_dataset import IamLinesDataset
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import torch.nn.functional as F

class HandwritingCenteringAndClipping:
    TEST_IMAGE_PATH = "/datastore/data/handwriting-recognition/IAM-database/data/lines/a01/a01-000u/a01-000u-00.png"

    def __init__(self):
        return

    @staticmethod
    def create_value_inverted_image_tensor(image_tensor: torch.Tensor):
        return torch.ones_like(image_tensor) - image_tensor

    @staticmethod
    def compute_unsmoothed_ink_density_for_image(image_tensor: torch.Tensor):
        print("image_tensor: " + str(image_tensor))
        image_tensor_float_format = \
            IamLinesDataset.convert_unsigned_int_image_tensor_to_float_image_tensor(image_tensor)
        print("image_tensor_float_format: " + str(image_tensor_float_format))
        value_inverted_image_tensor = HandwritingCenteringAndClipping.create_value_inverted_image_tensor(
            image_tensor_float_format)
        print("value inverted image tensor: " + str(value_inverted_image_tensor))
        util.image_visualization.imshow_tensor_2d(value_inverted_image_tensor.cpu())
        print("value_inverted_image_tensor.size():" + str(value_inverted_image_tensor.size()))
        result = torch.sum(value_inverted_image_tensor, 1)
        print("result.size(): " + str(result.size()))
        print("result: " + str(result))

        return result

    @staticmethod
    def get_smoothed_ink_density_center_index(image_tensor: torch.Tensor):
        unsmoothed_ink_density = HandwritingCenteringAndClipping.\
            compute_unsmoothed_ink_density_for_image(image_tensor)
        # HandwritingCenteringAndClipping.plot_density(unsmoothed_ink_density)
        # HandwritingCenteringAndClipping.plot_histogram(result)
        smoothed_density = HandwritingCenteringAndClipping.smooth_density(unsmoothed_ink_density)
        # HandwritingCenteringAndClipping.plot_density(smoothed_density)
        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528
        max, index =  torch.max(smoothed_density, 0)
        print("row index highest ink density: " + str(index))
        image_tensor_with_marked_center = image_tensor.clone()
        image_tensor_with_marked_center[index, :] = 0
        print("The marked ink density center:")
        util.image_visualization.imshow_tensor_2d(image_tensor_with_marked_center.cpu())


    @staticmethod
    def smooth_density(x: torch.Tensor):
        # See: https://discuss.pytorch.org/t/any-way-to-apply-gaussian-smoothing-on-tensor/13842
        # Create gaussian kernels
        kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]])
        # Apply smoothing
        x_3d = x.unsqueeze(0)
        x_3d = x_3d.unsqueeze(0)
        x_smooth = F.conv1d(x_3d, kernel)
        x_smooth = x_smooth.squeeze(0)
        x_smooth = x_smooth.squeeze(0)

        print("Before Smoothing: ")
        plt.plot(x.numpy())
        plt.show()
        print("After Smoothing: ")
        plt.plot(x_smooth.numpy())
        plt.show()
        return x_smooth

    @staticmethod
    def plot_density(one_dimensional_tensor: torch.Tensor):
        util.image_visualization.imshow_tensor_2d(one_dimensional_tensor.unsqueeze(0).cpu())

    @staticmethod
    def plot_histogram(one_dimensional_tensor: torch.Tensor):
        one_dimensional_tensor_numpy = one_dimensional_tensor.numpy()
        # See: https://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html
        n, bins, patches = plt.hist(one_dimensional_tensor, 50, density=1, facecolor='blue', alpha=0.75)
        plt.show()

    @staticmethod
    def get_test_image():
        test_image_numpy = io.imread(HandwritingCenteringAndClipping.TEST_IMAGE_PATH)
        test_image = torch.from_numpy(test_image_numpy)
        # util.image_visualization.imshow_tensor_2d(test_image.cpu())
        return test_image


def main():
    # test_iam_lines_dataset()
    HandwritingCenteringAndClipping.get_smoothed_ink_density_center_index(
        HandwritingCenteringAndClipping.get_test_image())


if __name__ == "__main__":
    main()
