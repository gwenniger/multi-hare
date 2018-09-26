import sys


class LinesWithExplicitWordSeparatorFileCreator:
    EXPLICIT_WORD_SEPARATOR_SYMBOL = "|"

    def __init__(self, input_file_path: str,
                 output_file_path: str,
                 explicit_word_separator_symbol: str,
                 integrate_word_separator_with_words: bool
                 ):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.explicit_word_separator_symbol = explicit_word_separator_symbol
        self.integrate_word_separator_with_words = integrate_word_separator_with_words

    @staticmethod
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    @staticmethod
    def create_lines_with_explicit_word_separator_file_creator(input_file_path: str,
                                                               output_file_path: str,
                                                               integrate_word_separator_with_words_string: str):

        integrate_word_separator_with_words = LinesWithExplicitWordSeparatorFileCreator.str2bool(
            integrate_word_separator_with_words_string)

        return LinesWithExplicitWordSeparatorFileCreator(
            input_file_path, output_file_path,
            LinesWithExplicitWordSeparatorFileCreator.EXPLICIT_WORD_SEPARATOR_SYMBOL,
            integrate_word_separator_with_words
        )

    def create_lines_with_explicit_word_separator_file(self):
        if self.integrate_word_separator_with_words:
            self.create_lines_with_explicit_integrated_word_separator_file()
        else:
            self.create_lines_with_explicit_separate_word_separator_file()

    def create_lines_with_explicit_separate_word_separator_file(self):
        with open(self.input_file_path, 'r') as file:
            with open(self.output_file_path, 'w') as output_file:
                for line in file:
                    output_line = ""
                    line_stripped = line.strip()
                    if len(line_stripped) > 0:
                        words = line_stripped.split()
                        if len(words) < 1:
                            raise RuntimeError("Error: words: " + str(words))
                        for word in words[0:len(words) - 1]:
                            output_line += word + " " + self.explicit_word_separator_symbol + " "
                        # Add the last word
                        word = words[len(words) - 1]
                        output_line += word
                        output_file.write(output_line + "\n")

                output_file.close()

    def create_lines_with_explicit_integrated_word_separator_file(self):
        with open(self.input_file_path, 'r') as file:
            with open(self.output_file_path, 'w') as output_file:
                for line in file:
                    output_line = ""
                    line_stripped = line.strip()
                    if len(line_stripped) > 0:
                        words = line_stripped.split()
                        if len(words) < 1:
                            raise RuntimeError("Error: words: " + str(words))
                        word = self.explicit_word_separator_symbol + words[0]
                        output_line += word
                        for word in words[1:len(words)]:
                            output_line += " " + self.explicit_word_separator_symbol + word
                        output_file.write(output_line + "\n")

                output_file.close()


def main():

    if len(sys.argv) != 4:
        raise RuntimeError("Error - usage: "
                           "lines_with_explicit_word_separator_file_creator INPUT_FILE_PATH OUTPUT_FILE_PATH "
                           "INTEGRATE_WORD_SEPARATOR_WITH_WORDS")

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    integrate_word_separator_with_words_string = sys.argv[3]
    lines_with_explicit_word_separator_file_creator = LinesWithExplicitWordSeparatorFileCreator.\
        create_lines_with_explicit_word_separator_file_creator(
            input_file_path, output_file_path, integrate_word_separator_with_words_string)
    lines_with_explicit_word_separator_file_creator.create_lines_with_explicit_word_separator_file()


if __name__ == "__main__":
    main()
