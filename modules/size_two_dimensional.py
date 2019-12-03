
__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class SizeTwoDimensional:

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    @staticmethod
    def create_size_two_dimensional(height: int, width: int):
        return SizeTwoDimensional(height, width)

    def __str__(self):
        result = "SizeTwoDimensional(" + str(self.height) + "," + str(self.width) + ")"
        return result

    def __repr__(self):
        return self.__str__()
