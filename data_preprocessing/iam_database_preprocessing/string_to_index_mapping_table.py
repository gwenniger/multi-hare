

class StringToIndexMappingTable:

    BLANK_SYMBOL = "<BLANK>"

    def __init__(self, string_to_index_map: dict, index_to_string_table: list, last_added_index: int):
        self.string_to_index_map = string_to_index_map
        self.index_to_string_table = index_to_string_table
        self.last_added_index = last_added_index

    def __str__(self):
        result = "<StringToIndexMappingTable>" + "\n"
        result += "number of elements: " + str(len(self.index_to_string_table)) + "\n"
        result += "</StringToIndexMappingTable>"
        return result

    @staticmethod
    def create_string_to_index_mapping_table():
        string_to_index_map =  dict([])
        index_to_string_table =  list([])
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



