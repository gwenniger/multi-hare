

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
