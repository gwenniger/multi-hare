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

        #print("list(image_tensor.size()): " + str(list(image_tensor.size())))
        # See: https://discuss.pytorch.org/t/indexing-a-2d-tensor/1667/2
        height = image_tensor.size(1)
        width = image_tensor.size(2)

        transformed_image = ImageInputTransformer.create_transformed_image(height, width, image_tensor)

        return transformed_image

    @staticmethod
    def create_transformed_image(height: int, width: int, image_tensor):
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
                new_row = torch.cat((torch.zeros(leading_zeros), image_tensor[0][y][:]), 0)
            else:
                new_row = image_tensor[0][y][:]
            #print("new row: " + str(new_row))
            new_row = torch.cat((new_row, torch.zeros(tailing_zeros)), 0)
            #print("new row: " + str(new_row))
            #print("transformed_image[0][y][:] : " + str(transformed_image[0][y][:]))
            transformed_image[0][y][:] = new_row[:]
            #for x in range(image_tensor.size(2)):
            #    # The transformed_image i'th row is shifted by i positions
            #    # print("image_tensor[0][x][y]: " + str(image_tensor[0][y][x]))
            #    # print("x: " + str(x) + " y: " + str(y))
            #    print("image_tensors: " + str(image_tensor))
            #    print("image_tensor[0][0][0]: " + str(image_tensor[0][0][0]))
            #    print("transformed_image[0][y][x + y]" + str( transformed_image[0][y][x + y]))
            #    transformed_image[0][y][x + y] = image_tensor[0][y][x]
        return transformed_image

    @staticmethod
    def create_row_diagonal_offset_tensors(image_tensors):
        image_tensor = image_tensors[0, :, :, :]
        #print("image_tensor: " + str(image_tensor))
        result = ImageInputTransformer.create_row_diagonal_offset_tensor(image_tensor)
        number_of_tensors = image_tensors.size(0)
        for i in range(1, number_of_tensors):
            #print("image number: " + str(i))
            skewed_image = ImageInputTransformer.create_row_diagonal_offset_tensor(image_tensors[i, :, :, :])
            result = torch.cat((result, skewed_image), 0)
        #print("result : " + str(result))
        return result
