import torch
import torch.nn as nn


class FullyConnectedLayersSharingWeights(torch.nn.Module):
    """
        This class implements fully connected layers with shared weights across directions.
        The motivation is that forcing the weight to be the same forces the inputs across
        directions to have similar structure. This is a form of structural bias / regularization
        that could make learning easier. Applied to the last MDLSTM layers, these forces these
        layers to use the same channels to indicate the same type of information, pushing
        the "synchrnonization" between the MDLSTM layers, rather than having only a weak
        synchronization with unique weights for every direction in the fully connected layers.

    """

    def __init__(self, number_of_input_channels_per_direction: int, number_of_output_channels: int,
                 number_of_directions: int):
            super(FullyConnectedLayersSharingWeights, self).__init__()
            self.number_of_input_channels_per_direction = number_of_input_channels_per_direction
            self.number_of_output_channels = number_of_output_channels
            self.number_of_directions = number_of_directions

            self.linear_layer = \
                nn.Linear(number_of_input_channels_per_direction, number_of_output_channels)

    @staticmethod
    def create_fully_connected_layers_sharing_weights(number_of_input_channels_per_direction: int,
                                                      number_of_output_channels: int,
                                                      number_of_directions: int):
        print("create_fully_connected_layers_sharing_weights - these layers share the same "
              "weights for every direction: " +
              "\n number_of_input_channels_per_direction: " + str(number_of_input_channels_per_direction) +
              "\n number_of_output_channels: " + str(number_of_output_channels) +
              "\n number_of_directions: " + str(number_of_directions))

        return FullyConnectedLayersSharingWeights(number_of_input_channels_per_direction, number_of_output_channels,
                                                  number_of_directions)

    def forward(self, input_activations_resized_two_dimensional):
        """
        :param input_activations_resized_two_dimensional:
        2-dimensional, with the zeroth dimension
        # the number of activations and the first dimension the number of channels

        :return: the activations
        """

        if self.number_of_directions > 1:
            # print("input_activations_resized_two_dimensional.size(): " +\
            #       str(input_activations_resized_two_dimensional.size()))
            # split into chunk along the channels dimension
            chunks = torch.chunk(input_activations_resized_two_dimensional, 4, 1)
            # append the chunks along the examples dimension
            chunks_appended = torch.cat(chunks, 0)
            # print("chunks_appended.size(): " + str(chunks_appended.size()))
            # compute the activations for the appended chunks
            activations = self.linear_layer(chunks_appended)
            # re-chunk the activations
            activations_chunked = torch.chunk(activations, 4, 0)
            # print("activations.size(): " + str(activations.size()))
            # for i, element in enumerate(activations_chunked):
            #     print("activations_chunked[" + str(i) + "].size()" + str(element.size()))
            # Stack the chunks to facilitate summation
            activations_chunked_stacked = torch.stack(activations_chunked, 0)
            # print("activations_chunked_stacked.size(): " + str(activations_chunked_stacked.size()))
            activations_chunked_stacked_summed = torch.sum(activations_chunked_stacked, 0)
            # print("activations_chunked_stacked_summed.size(): " + str(activations_chunked_stacked_summed.size()))
            # Remove bogus dimension 0 added for summation
            result = activations_chunked_stacked_summed.squeeze(0)
            # print("result.size(): " + str(result.size()))

        else:
            result = self.linear_layer(input_activations_resized_two_dimensional)

        return result
