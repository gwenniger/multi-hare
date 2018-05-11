import torch
from util.utils import Utils


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

        #print("list(image_tensor.size()): " + str(list(image_tensor.size())))
        # See: https://discuss.pytorch.org/t/indexing-a-2d-tensor/1667/2
        height = image_tensor.size(1)
        width = image_tensor.size(2)

        #number_of_image_tensors  = image_tensor.size(0)
        #print("number of image tensors: " + str(number_of_image_tensors))
        transformed_image = torch.zeros(1, height, (width * 2) - 1)
        #print("transformed_image: " + str(transformed_image))
        # print("transformed_image.size(): " + str(transformed_image.size()))

        for y in range(image_tensor.size(1)):
            leading_zeros = y
            tailing_zeros = transformed_image.size(2) - width - y
            #print("leading zeros: " + str(leading_zeros))
            #print("tailing_zeros: " + str(tailing_zeros))
            #print(" image_tensor[0][y][:]) : " + str( image_tensor[0][y][:]))
            if leading_zeros > 0:
                new_row = torch.cat((torch.zeros(leading_zeros), image_tensor[0, y, :]), 0)
            else:
                new_row = image_tensor[0, y][:]
            #print("new row: " + str(new_row))
            new_row = torch.cat((new_row, torch.zeros(tailing_zeros)), 0)
            #print("new row: " + str(new_row))
            #print("transformed_image[0][y][:] : " + str(transformed_image[0][y][:]))
            transformed_image[0, y, :] = new_row[:]
            #for x in range(image_tensor.size(2)):
            #    # The transformed_image i'th row is shifted by i positions
            #    # print("image_tensor[0][x][y]: " + str(image_tensor[0][y][x]))
            #    # print("x: " + str(x) + " y: " + str(y))
            #    print("image_tensors: " + str(image_tensor))
            #    print("image_tensor[0][0][0]: " + str(image_tensor[0][0][0]))
            #    print("transformed_image[0][y][x + y]" + str( transformed_image[0][y][x + y]))
            #    transformed_image[0][y][x + y] = image_tensor[0][y][x]
        return transformed_image

    # Non-optimized method, that computes the skewed images one at a time, then
    # concatenates them in a for loop
    @staticmethod
    def create_row_diagonal_offset_tensors_serial(image_tensors):
        image_tensor = image_tensors[0, :, :, :]
        result = ImageInputTransformer.create_row_diagonal_offset_tensor(image_tensor)
        number_of_tensors = image_tensors.size(0)
        for i in range(1, number_of_tensors):
        # print("image number: " + str(i))
            skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(image_tensors[i, :, :, :])
            result = torch.cat((result, skewed_image), 0)
        return result

    # Optimized method computes the complete set of skewed images all in one go
    # using pytorch tensor indexing to select slices of rows from multiple images
    # at one, doing the operation for all images in parallel
    # Requirement: all images must be of the same size
    @staticmethod
    def create_row_diagonal_offset_tensors_parallel(image_tensors):

        if Utils.use_cuda():
            # https://discuss.pytorch.org/t/which-device-is-model-tensor-stored-on/4908/7
            device = image_tensors.get_device()

        #See: https://stackoverflow.com/questions/46826218/pytorch-how-to-get-the-shape-of-a-tensor-as-a-list-of-int

        #print("list(image_tensor.size()): " + str(list(image_tensors.size())))
        # See: https://discuss.pytorch.org/t/indexing-a-2d-tensor/1667/2
        number_of_channels = image_tensors.size(1)
        height = image_tensors.size(2)
        width = image_tensors.size(3)

        number_of_image_tensors  = image_tensors.size(0)
        #print("number of image tensors: " + str(number_of_image_tensors))
        transformed_images = torch.zeros(number_of_image_tensors, number_of_channels, height, (width * 2) - 1)
        #print("transformed_image: " + str(transformed_image))
        # print("transformed_im   age.size(): " + str(transformed_image.size()))

        for y in range(image_tensors.size(2)):
            leading_zeros = y
            tailing_zeros = transformed_images.size(3) - width - y

            if leading_zeros > 0:

                # To get a sub-tensor with everything from the 0th and 3th dimension,
                # and specific values for the 1th  and 2nd dimension you use
                # image_tensors[:, 0, y, :]
                # See:
                # https://stackoverflow.com/questions/47374172/how-to-select-index-over-two-dimension-in-pytorch?rq=1
                leading_zeros_tensor = torch.zeros(number_of_image_tensors, number_of_channels,
                                                   leading_zeros)

                if Utils.use_cuda():
                    leading_zeros_tensor = leading_zeros_tensor.to(device)

                # print("leading_zeros_tensor.size()" + str(leading_zeros_tensor.size()))

                new_row = torch.cat((leading_zeros_tensor,
                                     image_tensors[:, :, y, :]), 2)
            else:
                new_row = image_tensors[:, :, y, :]

            if tailing_zeros > 0:
                # print("number of channels: " + str(number_of_channels))
                tailing_zeros_tensor = torch.zeros(number_of_image_tensors,
                                                   number_of_channels, tailing_zeros)
                if Utils.use_cuda():
                    tailing_zeros_tensor = tailing_zeros_tensor.to(device)

                # print("new_row.size(): " + str(new_row.size()))
                # print("tailing_zeros_tensor.size(): " + str(tailing_zeros_tensor.size()))
                new_row = torch.cat((new_row, tailing_zeros_tensor), 2)
            # print("new row.size(): " + str(new_row.size()))
            # print("transformed_image[:, :, y, :].size()" + str(transformed_images[:, :, y, :].size()))
            transformed_images[:, :, y, :] = new_row

        # print("transformed_images.size(): " + str(transformed_images.size()))
        # Remove the image channel dimension
        # transformed_images = transformed_images.squeeze(1)

        return transformed_images

    @staticmethod
    def create_row_diagonal_offset_tensors(image_tensors):

        #result = ImageInputTransformer.create_row_diagonal_offset_tensors_serial(image_tensors[:, :, :, :])
        result = ImageInputTransformer.create_row_diagonal_offset_tensors_parallel(image_tensors[:, :, :, :])
        #print("result: " + str(result))
        return result
