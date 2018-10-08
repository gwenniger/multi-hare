import sys
import re
from collections import OrderedDict


class LobOriginalPreprocessor:
    TEMPORARY_NEWLINE_MARKER = "<NEWLINE>"
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
                 keep_newlines_within_fragments: bool):
        self.lob_input_files_directory_path = lob_input_files_directory_path
        self.keep_newlines_within_fragments = keep_newlines_within_fragments

    @staticmethod
    def create_lob_original_preprocessor(lob_input_files_directory_path: str,
                                         keep_newlines_within_fragments: bool):
        return LobOriginalPreprocessor(lob_input_files_directory_path,
                                       keep_newlines_within_fragments)

    def perform_special_symbol_replacements(self, line: str):
        #result = line.replace("*<*", "")
        result = line.replace("**[SIC**]", "")
        result = re.sub("\*<\*[0-9]+\**\\\\*", "", result)
        result = re.sub("\\\\[0-9]+", "", result)
        result = result.replace("*<", "")
        result = result.replace("*>", "")
        result = result.replace("|^", LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER)
        result = result.replace("^", LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER)
        result = result.replace("**", "")
        result = re.sub("\*[0-9]+", "", result)
        result = re.sub("\*[0-9]+", "", result)
        result = result.replace("*", "")
        result = result.replace("\\", "")
        # Replace curly brackets
        result = re.sub("{[0-9]", "", result)
        result = re.sub("\"{", "\" ", result)
        result = re.sub("{", "", result)
        result = re.sub("}\"", " \"", result)
        result = re.sub("}", "", result)
        # Remove additional comments with square brackets
        result = re.sub("\[.*\]", "", result)
        return result

    @staticmethod
    def is_text_marking_contents_line(contents_line: str):
        return contents_line.startswith("**[") and contents_line.endswith("**]")

    @staticmethod
    def is_year_marking_contents_line(contents_line: str):
        return contents_line.startswith("*#")

    @staticmethod
    def contents_line_should_be_skipped(contents_line: str):
        return LobOriginalPreprocessor.is_text_marking_contents_line(contents_line) or \
               LobOriginalPreprocessor.is_year_marking_contents_line(contents_line)

    @staticmethod
    def is_title_contents_line(contents_line: str):
        if contents_line.startswith("*<") and contents_line.endswith("*>"):
            return True
        return False

    def process_lob_original_files(self):
        lob_part_id_to_lines_map = OrderedDict([])
        for lob_file_name in LobOriginalPreprocessor.LOB_ORIGINAL_FILE_NAMES:
            lob_origial_file_path = self.lob_input_files_directory_path + lob_file_name
            self.process_lob_original_file_lines(lob_origial_file_path, lob_part_id_to_lines_map)
        return lob_part_id_to_lines_map

    @staticmethod
    def perform_iam_specific_apostrophe_re_tokenization(line: str):
        """
        This function performs iam-specific apostrophe tokenization,
        to get certain apostrophe-containing constructions in the language model
        training data to be the same as in the iam dataset. Specifically
        iam uses at least the following set of somewhat alternative
        tokenizations:

        |‘ll|   (probably because it's short for "will")
        |‘ve|   (probably because it's short for "have")
        |‘d|    (probably because it's short for "would")
        |‘re|   (probably because it's short for "are")
        I|‘m|   (probably because it's short for "am")
        It|’s|  (probably because it's short for "is")
        That|’s| (probably because it's short for "is")
        she|‘s|  (probably because it's short for "is")


        :param line:
        :return:
        """
        result = line
        # Make "'ll" a separate token
        result = result.replace("'ll", " 'll")
        # Make "'ve" a separate token
        result = result.replace("'ve", " 've")
        # Make "'d" a separate token
        result = result.replace("'d", " 'd")
        # Make "'re" a separate token
        result = result.replace("'re", " 're")
        # Make "'m" a separate token
        result = result.replace("'m", " 'm")
        # Make "'s" in "It's" a separate token
        result = result.replace("It's", "It 's")
        # Make "'s" in "That's" a separate token
        result = result.replace("That's", "That 's")
        # Make "'s" in "she's" a separate token
        result = result.replace(" she's", " she 's")
        # Make "'s" in "he's" a separate token
        result = result.replace(" he's", " he 's")

        return result

    def perform_final_punctuation_correction(self, line: str):
        # print("perform_final_punctuation_correction - line: \n\"" +
        #      line + "\"\n")
        result = line
        # Use temporary newline markers to add whitespace before colon where appropriate
        result = re.sub("\.\s+" +
                        LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER,
                        " . " + LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER,
                        result)
        result = result.replace(".\" " + LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER,
                                " . \" " + LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER)
        # Remove remaining temporary newline markers
        if self.keep_newlines_within_fragments:
            result = re.sub("\s*" + LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER,
                            "\n",
                            result)
        else:
            result = result.replace(LobOriginalPreprocessor.TEMPORARY_NEWLINE_MARKER, "")
        # Add a whitespace before a comma, question mark,
        # exclamation sign and quotation sign to make them separate tokens
        result = result.replace(",", " ,")
        result = result.replace("?", " ?")
        result = result.replace("!", " !")
        result = result.replace("!\"", " ! \"")
        # https://stackoverflow.com/questions/3926451/how-to-match-but-not-capture-part-of-a-regex?rq=1
        # look ahead: next character is letter
        result = re.sub("\"(?=\w)", "\" ", result)
        # look behind: previous character is letter
        result = re.sub("(?<=\w)\"", " \"", result)
        # If the last character is a colon, add a whitespace in front of it
        result = result.strip()
        if result.endswith("."):
            result = result[0:len(result)-1] + " ."
        elif result.endswith(".\""):
            result = result[0:len(result) - 2] + " . \""
        # Replace comma followed by quotation
        result = result.replace(",\"", ", \"")

        # Do some apostrophe re-tokenization mimic the weird tokenization used in the
        # iam dataset of apostrophes in verbs
        result = LobOriginalPreprocessor.perform_iam_specific_apostrophe_re_tokenization(result)

        return result

    def process_lob_original_file_lines(self, lob_input_file_path: str, lob_part_id_to_lines_map: OrderedDict):

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
                            current_lines_list.append(
                                self.perform_final_punctuation_correction(
                                    current_line_accumulator))
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
                                current_lines_list.append(
                                    self.perform_final_punctuation_correction(
                                        current_line_accumulator)
                                )
                            current_lines_list.append(special_symbol_replaced_contents)
                            current_line_accumulator = ""
                        else:
                            current_line_accumulator += special_symbol_replaced_contents + " "

                else:
                    print("Skipping non-content line...")

    def process_lob_original_file_lines_and_write_to_file(self, output_file_path: str):

        with open(output_file_path, 'w') as output_file:
            lob_part_id_to_lines_map = self.process_lob_original_files()

            for part_id in lob_part_id_to_lines_map.keys():
                # output_file.write("\n<part-id-" + part_id + ">")
                lines = lob_part_id_to_lines_map[part_id]
                for line in lines:
                    output_file.write(line + "\n")
                # output_file.write("\n</part-id-" + part_id + ">")
            # output_file.close()


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Error: lob_original_preprocessor LOB_INPUT_FILES_DIRECTORY OUTPUT_FILE_PATH")

    lob_input_files_directory = sys.argv[1]
    output_file_path = sys.argv[2]
    lob_original_preprocessor = LobOriginalPreprocessor.create_lob_original_preprocessor(lob_input_files_directory)
    lob_original_preprocessor.process_lob_original_file_lines_and_write_to_file(output_file_path)


if __name__ == "__main__":
    main()
