import sys
from data_preprocessing.monolingual_data_preprocessing.comments_cleaned_lines_extractor \
    import CommentsCleanedTextExtractor


class CasedWordVersionRetriever:

    def __init__(self, cased_characters: list):
        self.cased_characters = cased_characters
        self.current_index = 0
        self.not_found_count = 0

    @staticmethod
    def create_cased_word_version_retriever(
            comments_cleaned_text_extractor: CommentsCleanedTextExtractor):
        cased_characters = comments_cleaned_text_extractor.get_non_whitespace_characters()
        return CasedWordVersionRetriever(cased_characters)

    def get_cased_word(self, begin_index, word):
        last_index = begin_index + len(word)
        cased_word_characters = self.cased_characters[begin_index:last_index]
        result = "".join(cased_word_characters)
        return result

    def retrieve_cased_version_word(self, word: str):

        result = self.get_cased_word(self.current_index, word)

        print("input: " + word + " result: " + result)
        if result.lower() == word.lower():
            self.current_index += len(word)
            return result
        else:
            # By allowing only words of length >= 2 to skip, the
            # chance to illegally skip to far forward is reduced
            if len(word) > 2:
                # Try skipping characters, but only if the word is sufficiently long
                # These skip numbers are experimentally chosen to
                # work on the LOB files.
                # If too long a jump is allowed, you risk skipping over
                # a section
                for characters_to_skip in range(1, 60):
                    result = self.get_cased_word(self.current_index + characters_to_skip,
                                                 word)
                    if result.lower() == word.lower():
                        self.current_index += len(word) + characters_to_skip
                        print("input: " + word + " result second attempt: " + result)
                        return result

            print("warning: did not find characters for \"" + word +
                  "\" - using the original")
            if str(word).isalnum():
                self.not_found_count += 1
            # This is a check that can be used for debugging
            # if self.not_found_count >= 10000:
            #    raise RuntimeError("Error")
            return word


class UntaggedTextExtractor:
    TAG_SEPARATOR_LOBS_CORPUS = "_"
    NEW_LINE_PREFIX_LOBS = " "

    def __init__(self, tagged_input_file_path: str,
                 output_file_path: str,
                 tag_separator: str,
                 new_line_prefix: str,
                 cased_word_version_retriever: CasedWordVersionRetriever
                 ):
        self.tagged_input_file_path = tagged_input_file_path
        self.output_file_path = output_file_path
        self.tag_separator = tag_separator
        self.new_line_prefix = new_line_prefix
        self.cased_word_version_retriever = cased_word_version_retriever

    @staticmethod
    def create_untagged_text_extractor_lobs(tagged_input_file_path, cased_input_file_path: str,
                                            output_file_path):

        comments_cleaned_text_extractor = CommentsCleanedTextExtractor.create_cleaned_text_extractor_lobs(cased_input_file_path)
        cased_word_version_retriever = CasedWordVersionRetriever.\
            create_cased_word_version_retriever(comments_cleaned_text_extractor)

        return UntaggedTextExtractor(tagged_input_file_path, output_file_path,
                                     UntaggedTextExtractor.TAG_SEPARATOR_LOBS_CORPUS,
                                     UntaggedTextExtractor.NEW_LINE_PREFIX_LOBS,
                                     cased_word_version_retriever)

    def get_word_from_word_tag_pair(self, word_tag_pair):
        return word_tag_pair.split(self.tag_separator)[0]

    def create_untagged_output_file(self):

        print("output file path: " + self.output_file_path)

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
                        cased_word = self.cased_word_version_retriever.retrieve_cased_version_word(word)

                        output_line += cased_word + " "
                    # Add the last word
                    word_tag_pair = word_tag_pairs[len(word_tag_pairs) - 1]
                    word = self.get_word_from_word_tag_pair(word_tag_pair)
                    cased_word = self.cased_word_version_retriever.retrieve_cased_version_word(word)
                    print("cased word: " + str(cased_word))
                    output_line += cased_word
                    if line.startswith(self.new_line_prefix):
                        output_file.write("\n" + output_line)
                    else:
                        output_file.write(" " + output_line )
                output_file.write("\n")
                output_file.close()


def main():

    if len(sys.argv) != 4:
        raise RuntimeError("Error: usage untagged_text_extractor TAGGED_INPUT_FILE_PATH "
                           "CASED_INPUT_FILE_PATH OUTPUT_FILE_PATH")

    tagged_input_file_path = sys.argv[1]
    cased_input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    untagged_text_extractor = UntaggedTextExtractor.create_untagged_text_extractor_lobs(
        tagged_input_file_path, cased_input_file_path, output_file_path)
    untagged_text_extractor.create_untagged_output_file()


if __name__ == "__main__":
    main()
