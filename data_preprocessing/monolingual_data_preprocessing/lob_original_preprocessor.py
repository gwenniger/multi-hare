import sys
import re
from collections import OrderedDict


class LobOriginalPreprocessor:
    LOB_ORIGINAL_FILE_NAMES = list(["LOB_A.TXT",
                                    "LOB_B.TXT",
                                    "LOB_C.TXT",
                                    "LOB_D.TXT",
                                    "LOB_E.TXT",
                                    "LOB_F.TXT",
                                    "LOB_G.TXT",
                                    "LOB_H.TXT",
                                    "LOB_J.TXT",
                                    "LOB_K.TXT",
                                    "LOB_L.TXT",
                                    "LOB_M.TXT",
                                    "LOB_N.TXT",
                                    "LOB_P.TXT",
                                    "LOB_R.TXT"])
    NUM_LOB_LINE_PREAMBLE_CHARACTERS = 8

    def __init__(self, lob_input_files_directory_path: str,
                 output_file_path: str):
        self.lob_input_files_directory_path = lob_input_files_directory_path
        self.output_file_path = output_file_path

    @staticmethod
    def create_lob_original_preprocessor(lob_input_files_directory_path: str,
                 output_file_path: str):
        return LobOriginalPreprocessor(lob_input_files_directory_path, output_file_path)

    def process_lob_original_files(self):
        with open(self.output_file_path, 'w') as output_file:

            for lob_file_name in LobOriginalPreprocessor.LOB_ORIGINAL_FILE_NAMES[0:1]:
                lob_origial_file_path = self.lob_input_files_directory_path + lob_file_name
                self.process_lob_original_file_lines(lob_origial_file_path, output_file)

    def perform_special_symbol_replacements(self, line: str):
        #result = line.replace("*<*", "")
        result = re.sub("\*<\*[0-9]+\**", "", line)
        result = re.sub("\\\\[0-9]+", "", result)
        result = result.replace("*<", "")
        result = result.replace("*>", "")
        result = result.replace("|^", "")
        result = result.replace("^", "")
        result = result.replace("**", "")
        result = re.sub("\*[0-9]+", "", result)
        result = re.sub("\*[0-9]+", "", result)
        result = re.sub("{", "", result)
        result = re.sub("}", "", result)
        result = result.replace("*", "")
        return result

    @staticmethod
    def is_text_marking_contents_line(contents_line: str):
        return contents_line.startswith("**[") and contents_line.endswith("**]")

    @staticmethod
    def contents_line_should_be_skipped(contents_line: str):
        return LobOriginalPreprocessor.is_text_marking_contents_line(contents_line)

    @staticmethod
    def is_title_contents_line(contents_line: str):
        if contents_line.startswith("*<") and contents_line.endswith("*>"):
            return True
        return False

    def process_lob_original_file_lines(self, lob_input_file_path: str, output_file):
        lob_part_id_to_lines_map = OrderedDict([])

        current_part_id = None
        current_lines_list = None
        current_line_accumulator = ""
        with open(lob_input_file_path, 'r') as file:
            line_index = 0
            for line in file:
                line_index += 1
                # if line_index > 1000:
                #    break
                print("line: \"" + str(line) + "\"")
                if line[0].isalpha():
                    part_id = line[0:3].strip().lower()

                    if part_id != current_part_id:

                        # Add the current lines list to the map
                        if current_lines_list is not None:
                            current_lines_list.append(current_line_accumulator)
                            lob_part_id_to_lines_map[current_part_id] = current_lines_list
                            current_line_accumulator = ""
                        current_lines_list = list([])
                        current_part_id = part_id
                    print("current_line_accumulator: \"" + current_line_accumulator + "\"")

                    line_contents_part = line[8:].strip()

                    print("line_contents_part: \"" + line_contents_part + "\"")

                    if not LobOriginalPreprocessor.contents_line_should_be_skipped(line_contents_part):

                        # Replace the special symbols in the contents part
                        special_symbol_replaced_contents = self.perform_special_symbol_replacements(line_contents_part)
                        print("special-symbol-replaced contents: \"" + special_symbol_replaced_contents + "\"")
                        if LobOriginalPreprocessor.is_title_contents_line(line_contents_part):
                            if len(current_line_accumulator) > 0:
                                current_lines_list.append(current_line_accumulator)
                            current_lines_list.append(special_symbol_replaced_contents)
                            current_line_accumulator = ""
                        else:
                            current_line_accumulator += special_symbol_replaced_contents + " "

                else:
                    print("Skipping non-content line...")

        for part_id in lob_part_id_to_lines_map.keys():
            output_file.write("\n<part-id-" + part_id + ">")
            lines = lob_part_id_to_lines_map[part_id]
            for line in lines:
                output_file.write("\n" + line)
            output_file.write("\n</part-id-" + part_id + ">")
        # output_file.close()


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Error: lob_original_preprocessor LOB_INPUT_FILES_DIRECTORY OUTPUT_FILE_PATH")

    lob_input_files_directory = sys.argv[1]
    output_file_path = sys.argv[2]
    lob_original_preprocessor = LobOriginalPreprocessor.create_lob_original_preprocessor(lob_input_files_directory,
                                                                                         output_file_path)
    lob_original_preprocessor.process_lob_original_files()


if __name__ == "__main__":
    main()
