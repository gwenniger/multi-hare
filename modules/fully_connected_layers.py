import torch
import torch.nn as nn


class FullyConnectedLayers(torch.nn.Module):

    def __init__(self, number_of_input_channels_per_direction: int, number_of_output_channels: int,
                 number_of_directions: int):
            super(FullyConnectedLayers, self).__init__()
            self.number_of_input_channels_per_direction = number_of_input_channels_per_direction
            self.number_of_output_channels = number_of_output_channels
            self.number_of_directions = number_of_directions

            self.one_dimensional_grouped_convolution = \
                nn.Conv1d(number_of_input_channels_per_direction * number_of_directions,
                          number_of_output_channels * number_of_directions, 1,
                          stride=1, groups=number_of_directions, bias=True)

    @staticmethod
    def create_fully_connected_layers(number_of_input_channels_per_direction: int,
                                      number_of_output_channels: int,
                                      number_of_directions: int):
        print("create_fully_connected_layers: " +
              "\n number_of_input_channels_per_direction: " + str(number_of_input_channels_per_direction) +
              "\n number_of_output_channels: " + str(number_of_output_channels) +
              "\n number_of_directions: " + str(number_of_directions))

        return FullyConnectedLayers(number_of_input_channels_per_direction, number_of_output_channels,
                                    number_of_directions)

    def forward(self, input_activations_resized_two_dimensional):
        """
        :param input_activations_resized_two_dimensional:
        2-dimensional, with the zeroth dimension
        # the number of activations and the first dimension the number of channels

        :return: the activations
        """

        # print("fully_connected_layers.forward -  input_activations_resized_two_dimensional.size(): " +
        #       str(input_activations_resized_two_dimensional.size()))
        # The 1-d convolution layer expects the first dimension to be channels and the second examples
        input_activations_resized_two_dimensional_transposed =\
            input_activations_resized_two_dimensional.transpose(0, 1)
        # A batch size is required, because the 1-d convolution expects the zeroth dimension to be
        # the batch size
        input_activations_resized_two_dimensional_transposed = \
            input_activations_resized_two_dimensional_transposed.unsqueeze(0)

        activations = self.one_dimensional_grouped_convolution(input_activations_resized_two_dimensional_transposed)

        if self.number_of_directions > 1:
            # print("FullyConnectedLayers.forward - input_activations_resized_two_dimensional.size(): "
            #       + str(input_activations_resized_two_dimensional.size()))
            # print("FullyConnectedLayers.forward - activations.size(): "
            #       + str(activations.size()))
            # Retrieve the outputs for the different directions
            activations_per_direction = torch.chunk(activations, self.number_of_directions, 1)
            # If there are multiple directions, the results over the different directions are summed
            # Stack the outputs for the different directions
            stacked_activations = torch.stack(activations_per_direction, 0)
            # print("stacked_activations.size() : " + str(stacked_activations.size()))
            result = torch.sum(stacked_activations, 0)
            # print("FullyConnectedLayers.forward - result.size(): " + str(result.size()))
        else:
            result = activations

        # The expected output format is (bogus) batch size, examples-length, output channels
        # therefore, the first and second dimension need to be swapped
        result = result.transpose(1, 2)

        return result
