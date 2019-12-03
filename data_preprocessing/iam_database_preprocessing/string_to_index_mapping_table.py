
__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class StringToIndexMappingTable:

    #    BLANK_SYMBOL = "<BLANK>"
    # BLANK_SYMBOL = "_"
    # This symbol should not be used in the annotation, that is why the obscure
    # "␣" symbol is used rather than the "_" symbol which still appears in some datasets
    # See also https://en.wikipedia.org/wiki/Whitespace_character#Substitutes
    BLANK_SYMBOL = "␣"
    WORD_KEY_SEPARATOR = "|||"

    def __init__(self, string_to_index_map: dict, index_to_string_table: list, last_added_index: int):
        self.string_to_index_map = string_to_index_map
        self.index_to_string_table = index_to_string_table
        self.last_added_index = last_added_index

    def __str__(self):
        result = "<StringToIndexMappingTable>" + "\n"
        result += "number of elements: " + str(len(self.index_to_string_table)) + "\n"
        for index in range(0, len(self.index_to_string_table)):
            result += "\nindex: " + str(index) + " => \"" + self.index_to_string_table[index] + "\""
        result += "\n</StringToIndexMappingTable>"
        return result

    @staticmethod
    def create_string_to_index_mapping_table():
        string_to_index_map = dict([])
        index_to_string_table = list([])
        last_added_index = -1
        result = StringToIndexMappingTable(string_to_index_map, index_to_string_table, last_added_index)
        # The blank symbol is added at at the beginning
        result.add_string(StringToIndexMappingTable.BLANK_SYMBOL)
        return result

    def get_string(self, index: int):
        return self.index_to_string_table[index]

    def get_index(self, string: str):
        return self.string_to_index_map[string]

    def get_indices(self, strings: list):
        result = list([])
        for string in strings:
            result.append(self.get_index(string))
        return result

    def add_string(self, string: str):
        if string in self.string_to_index_map:
            return
        else:
            self.last_added_index += 1
            self.string_to_index_map[string] = self.last_added_index
            self.index_to_string_table.append(string)
            return

    def add_strings(self, strings: list):
        for string in strings:
            self.add_string(string)

    def get_vocabulary_list(self):
        return list(self.index_to_string_table)

    @staticmethod
    def get_blank_symbol():
        return StringToIndexMappingTable.BLANK_SYMBOL

    def save_string_to_index_mapping_table_to_file(self, table_output_file_path: str):
        print("Saving string to index mapping table to file: " + table_output_file_path + " ...")
        with open(table_output_file_path, "w") as output_file:
            for index in range(0, len(self.index_to_string_table)):
                output_file.write(str(index) + StringToIndexMappingTable.WORD_KEY_SEPARATOR + self.index_to_string_table[index] + "\n")
            output_file.close()
        print("Done")

    @staticmethod
    def read_string_to_index_mapping_table_from_file(input_file_path):
        print("Reading permutation from input file " + input_file_path + " ...")

        string_to_index_map = dict([])
        index_to_string_table = list([])
        last_added_index = -1
        result = StringToIndexMappingTable(string_to_index_map, index_to_string_table, last_added_index)

        with open(input_file_path) as f:
            content = f.readlines()

            # https://stackoverflow.com/questions/12330522/reading-a-file-without-newlines
            new_line_stripped_lines = [line.rstrip('\n') for line in content]

            line_index = 0
            for line in new_line_stripped_lines:
                if len(line.strip()) > 0:

                    parts = line.split(StringToIndexMappingTable.WORD_KEY_SEPARATOR)
                    print("read_string_to_index_mapping_table_from_file - parts: " + str(parts))
                    # print("line: " + line)
                    index = int(parts[0])

                    if not(line_index == index):
                        raise RuntimeError("RuntimeError: unexpected index " + str(index) +
                                           " at line " + str(line_index) +
                                           " , expected one index and one word per line, starting from index 0")

                    word = parts[1]
                    print("read_string_to_index_mapping_table_from_file - word: " + str(word))

                    result.add_string(word)
                    line_index += 1

        return result


