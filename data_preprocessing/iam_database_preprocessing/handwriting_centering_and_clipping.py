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
    #TEST_IMAGE_PATH = "/datastore/data/handwriting-recognition/IAM-database/data/lines/a01/a01-000u/a01-000u-00.png"
    # TEST_IMAGE_PATH = "/datastore/data/handwriting-recognition/IAM-database/data/lines/a01/a01-000u/a01-000u-06.png"
    # TEST_IMAGE_PATH = "/datastore/data/handwriting-recognition/IAM-database/data/" \
    #                   "sentences/n03/n03-120/n03-120-s00-00.png"
    TEST_IMAGE_PATH = "/datastore/data/handwriting-recognition/IAM-database/data/" \
                      "sentences/n03/n03-120/n03-120-s00-04.png"
#

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
    def find_first_index_with_higher_or_equal_value(one_dimensional_tensor: torch.Tensor,
                                                    value):
        for i in range(0, one_dimensional_tensor.size(0)):
            if one_dimensional_tensor[i] >= value:
                return i
        return -1

    @staticmethod
    def find_first_index_with_higher_or_equal_value_back_to_front(one_dimensional_tensor: torch.Tensor,
                                                                  value):
        for i in range(one_dimensional_tensor.size(0) - 1, 0, -1):
            if one_dimensional_tensor[i] >= value:
                return i

    @staticmethod
    def get_smoothed_ink_density_center_index(image_tensor: torch.Tensor):
        unsmoothed_ink_density = HandwritingCenteringAndClipping.\
            compute_unsmoothed_ink_density_for_image(image_tensor)
        # HandwritingCenteringAndClipping.plot_density(unsmoothed_ink_density)
        # HandwritingCenteringAndClipping.plot_histogram(result)
        smoothed_density = HandwritingCenteringAndClipping.smooth_density(unsmoothed_ink_density)
        # HandwritingCenteringAndClipping.plot_density(smoothed_density)
        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528
        smoothed_ink_density_max, index = torch.max(smoothed_density, 0)
        print("highest ink density: " + str(smoothed_ink_density_max))
        print("row index highest ink density: " + str(index))
        image_tensor_with_marked_center = image_tensor.clone()
        image_tensor_with_marked_center[index, :] = 0
        print("The marked ink density center:")
        util.image_visualization.imshow_tensor_2d(image_tensor_with_marked_center.cpu())

        smoothed_density_mean = np.mean(smoothed_density.numpy())
        smoothed_density_std = np.std(smoothed_density.numpy())
        print("Smoothed density standard mean: " + str(smoothed_density_mean))
        print("Smoothed density standard deviation: " + str(smoothed_density_std))

        min_accepted_ink_density = smoothed_ink_density_max - 2.9 * smoothed_density_std

        clipping_bottom_index = HandwritingCenteringAndClipping.\
            find_first_index_with_higher_or_equal_value(smoothed_density, min_accepted_ink_density)
        clipping_top_index = HandwritingCenteringAndClipping. \
            find_first_index_with_higher_or_equal_value_back_to_front(smoothed_density, min_accepted_ink_density)

        image_tensor_with_marked_center_and_clipping = image_tensor_with_marked_center.clone()
        image_tensor_with_marked_center_and_clipping[clipping_bottom_index, :] = 0
        image_tensor_with_marked_center_and_clipping[clipping_top_index, :] = 0
        print("The marked clipping borders:")
        util.image_visualization.imshow_tensor_2d(image_tensor_with_marked_center_and_clipping.cpu())
        print("The clipped image:")
        clipped_image = image_tensor[clipping_bottom_index:clipping_top_index, :]
        util.image_visualization.imshow_tensor_2d(clipped_image.cpu())


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
