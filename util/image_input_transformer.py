import torch

class ImageInputTransformer:


    # This method takes an image and creates a transformed image, shifting the i-th row
    # with i pixels. This corresponds to the transformation used in the
    # pixel recurrent neural networks paper (https://arxiv.org/pdf/1601.06759.pdf)
    # This trick can be used to efficiently compute Multi-dimensional RNNs, while
    # keeping the input the same for every layer of the network
    #
    @staticmethod
    def create_row_diagonal_offset_tensor(image_tensor):
        #See: https://stackoverflow.com/questions/46826218/pytorch-how-to-get-the-shape-of-a-tensor-as-a-list-of-int

        # print("list(image_tensor.size()): " + str(list(image_tensor.size())))
        # See: https://discuss.pytorch.org/t/indexing-a-2d-tensor/1667/2
        height = image_tensor.size(1)
        width = image_tensor.size(2)
        transformed_image = torch.zeros(1, height, (width * 2) - 1)
        # print("transformed_image: " + str(transformed_image))
        # print("transformed_image.size(): " + str(transformed_image.size()))

        for y in range(image_tensor.size(1)):
            for x in range(image_tensor.size(2)):
                # The transformed_image i'th row is shifted by i positions
                # print("image_tensor[0][x][y]: " + str(image_tensor[0][y][x]))
                # print("x: " + str(x) + " y: " + str(y))
                transformed_image[0][y][x+y] = image_tensor[0][y][x]

        return transformed_image

    @staticmethod
    def create_row_diagonal_offset_tensors(image_tensors):
        image_tensor = image_tensors[0, :, :, :]
        # print("image_tensor: " + str(image_tensor))
        result = ImageInputTransformer.create_row_diagonal_offset_tensor(image_tensor)
        number_of_tensors = image_tensors.size(0)
        for i in range(1, number_of_tensors):
            skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(image_tensors[i, :, :, :])
            torch.cat((result, skewed_image), 0)
        return result
