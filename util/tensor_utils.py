import torch
import util.image_visualization

class TensorUtils:

    @staticmethod
    def tensors_are_equal(tensor_one, tensor_two):
        # https://stackoverflow.com/questions/32996281/how-to-check-if-two-torch-tensors-or-matrices-are-equal
        # https://discuss.pytorch.org/t/tensor-math-logical-operations-any-and-all-functions/6624
        return torch.eq(tensor_one, tensor_two).all()

    # Debugging method that checks that two lists of (3-dimensional)
    # tensors are equal, visualizing the first pair of tensors it
    # encounters that is not equal, if there is one
    @staticmethod
    def tensors_lists_are_equal(tensor_list_one, tensor_list_two):
        if len(tensor_list_one) != len(tensor_list_two):
            return False

        index = 0
        for tensor_one, tensor_two in zip(tensor_list_one, tensor_list_two):
            if not TensorUtils.tensors_are_equal(tensor_one, tensor_two):
                print("tensor_lists_are_equal --- \n"
                      "tensor_list_one[" + str(index) + "]: \n"  +
                      str(tensor_one) + "\n" + "and " +
                      "tensor_list_two[" + str(index) + "]" +
                      str(tensor_two) + " are not equal.")

                print("showing tensor one:")
                element_without_channel_dimension = tensor_one.squeeze(0)
                util.image_visualization.imshow_tensor_2d(element_without_channel_dimension)

                print("showing tensor two:")
                element_without_channel_dimension = tensor_two.squeeze(0)
                util.image_visualization.imshow_tensor_2d(element_without_channel_dimension)

                return False
            index += 1
        return True

    @staticmethod
    # Debugging method that finds equal slices in a 4-dimensional tensor over the batch dimension
    # and visualizes them
    def find_equal_slices_over_batch_dimension(tensor):
        number_or_slices = tensor.size(0)

        for slice_one_index in range(0, number_or_slices):
            tensor_slice_one = tensor[slice_one_index, :, :, :]

            for slice_two_index in range(slice_one_index + 1, number_or_slices):
                tensor_slice_two = tensor[slice_two_index, :, :, :]

                tensors_are_equal = TensorUtils.tensors_are_equal(tensor_slice_one, tensor_slice_two)

                if tensors_are_equal:
                    print("find_equal_slices_over_batch_dimension --- \n"
                          "tensor[" + str(slice_one_index) + ",:,:,:]: \n" +
                          str(tensor_slice_one) + "\n" + "and " +
                          "tensor[" + str(slice_two_index) + "]" +
                          str(tensor_slice_two) + " are equal.")

                    print("showing tensor slice one:")
                    element_without_channel_dimension = tensor_slice_one.squeeze(0)
                    util.image_visualization.imshow_tensor_2d(element_without_channel_dimension)

                    print("showing tensor slice two:")
                    element_without_channel_dimension = tensor_slice_two.squeeze(0)
                    util.image_visualization.imshow_tensor_2d(element_without_channel_dimension)


    @staticmethod
    def number_of_zeros(tensor):
        mask = tensor.eq(0)
        zero_elements = torch.masked_select(tensor, mask).view(-1)
        number_of_zeros = zero_elements.size(0)
        return number_of_zeros

    @staticmethod
    def number_of_ones(tensor):
        mask = tensor.eq(1)
        one_elements = torch.masked_select(tensor, mask).view(-1)
        number_of_ones = one_elements.size(0)
        return number_of_ones

    @staticmethod
    def number_of_non_ones(tensor):
        mask = tensor.eq(1)
        one_elements = torch.masked_select(tensor, mask).view(-1)
        number_of_elements = tensor.view(-1).size(0)
        number_of_ones = one_elements.size(0)
        return number_of_elements - number_of_ones

    @staticmethod
    def sum_list_of_tensors(list_of_tensors):
        result = list_of_tensors[0]

        for index in range(1, len(list_of_tensors)):
            # if TensorUtils.tensors_are_equal(result, list_of_tensors[index]):
            #     print("WARNING - sum_list_of_tensors - tensors are equal")
            # else:
            #     print("INFO - sum_list_of_tensors - tensors are not equal")

            # print("result before addition: " + str(result))
            # print("to add: list_of_tensors[" + str(index) + "]:" + str(list_of_tensors[index]))
            result += list_of_tensors[index]
            # print("result after addition: " + str(result))
        return result
