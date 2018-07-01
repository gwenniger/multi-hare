import numpy
import os.path

class DataPermutation:

    def __init__(self, permutation, permutation_output_file_path):
        self.permutation = permutation
        self.permutation_output_file_path = permutation_output_file_path
        return

    @staticmethod
    def create_data_permutation(permutation_length, permutation_output_file_path):
        print("Creating new permutation...")
        lines_permutation = numpy.random.permutation(permutation_length)
        return DataPermutation(lines_permutation, permutation_output_file_path)

    @staticmethod
    def create_data_permutation_from_saved_permutation(permutation_input_file_path):
        print("Create data permutation from saved permutation file...")
        lines_permutation = DataPermutation.read_permutation_from_file(permutation_input_file_path)
        return DataPermutation(lines_permutation, None)

    @staticmethod
    def load_or_create_and_save_permutation(permutation_length,
                                            permutation_save_or_load_file_path: str):

        if os.path.isfile(permutation_save_or_load_file_path):
            print("load_or_create_and_save_permutation - file exists, loading...")
            data_permutation = DataPermutation. \
                create_data_permutation_from_saved_permutation(permutation_save_or_load_file_path)
            if not len(data_permutation.permutation) == permutation_length:
                raise RuntimeError("Error: loaded permutation is not of the right length")
        else:
            data_permutation = DataPermutation.create_data_permutation(permutation_length,
                                                                       permutation_save_or_load_file_path)
            print("Saving the permutation to \"" + permutation_save_or_load_file_path +
                  "\" for future use...")
            data_permutation.save_permutation_order_file()
        return data_permutation

    @staticmethod
    def read_permutation_from_file(input_file_path):
        print("Reading permutation from input file " + input_file_path + " ...")
        permutation = []
        with open(input_file_path) as f:
            content = f.readlines()
            for line in content:
                # print("line: " + line)
                permutation.append(int(line.strip()))
        print("Read permutation...")
        print("Showing last 100 elements for checking...")
        j = len(permutation) - 100
        k = len(permutation)
        for i in range(j, k):
            print("Permutation element[" + str(i) + "]:" + str(permutation[i]))

        return permutation

    def save_permutation_order_file(self):
        print("Creating permutation order file " + self.permutation_output_file_path + " ...")
        with open(self.permutation_output_file_path, "w") as output_file:
            for index in self.permutation:
                output_file.write(str(index) + "\n")
            output_file.close()
        print("Done")

