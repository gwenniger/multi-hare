import sys


class UntaggedTextExtractor:
    TAG_SEPARATOR_LOBS_CORPUS = "_"
    NEW_LINE_PREFIX_LOBS = " "

    def __init__(self, tagged_input_file_path: str,
                 output_file_path: str,
                 tag_separator: str,
                 new_line_prefix: str,
                 ):
        self.tagged_input_file_path = tagged_input_file_path
        self.output_file_path = output_file_path
        self.tag_separator = tag_separator
        self.new_line_prefix = new_line_prefix

    @staticmethod
    def create_untagged_text_extractor_lobs(tagged_input_file_path, output_file_path):
        return UntaggedTextExtractor(tagged_input_file_path, output_file_path,
                                     UntaggedTextExtractor.TAG_SEPARATOR_LOBS_CORPUS,
                                     UntaggedTextExtractor.NEW_LINE_PREFIX_LOBS)

    def get_word_from_word_tag_pair(self, word_tag_pair):
        return word_tag_pair.split(self.tag_separator)[0]

    def create_untagged_output_file(self):
        with open(self.tagged_input_file_path, 'r') as file:
            with open(self.output_file_path, 'w') as output_file:
                for line in file:
                    output_line = ""
                    line_stripped = line.strip()
                    # The str.split() method without an argument splits on whitespace
                    # https://stackoverflow.com/questions/8113782/split-string-on-whitespace-in-python
                    word_tag_pairs = line_stripped.split()
                    for word_tag_pair in word_tag_pairs[0:len(word_tag_pairs) - 1]:
                        word = self.get_word_from_word_tag_pair(word_tag_pair)
                        output_line += word + " "
                    # Add the last word
                    word_tag_pair = word_tag_pairs[len(word_tag_pairs) - 1]
                    word = self.get_word_from_word_tag_pair(word_tag_pair)
                    output_line += word
                    if line.startswith(self.new_line_prefix):
                        output_file.write("\n" + output_line)
                    else:
                        output_file.write(" " + output_line )
                output_file.write("\n")
                output_file.close()


def main():

    if len(sys.argv) != 3:
        raise RuntimeError("Error: usage untagged_text_extractor TAGGED_INPUT_FILE_PATH OUTPUT_FILE_PATH")

    tagged_input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    untagged_text_extractor = UntaggedTextExtractor.create_untagged_text_extractor_lobs(
        tagged_input_file_path, output_file_path)
    untagged_text_extractor.create_untagged_output_file()


if __name__ == "__main__":
    main()
