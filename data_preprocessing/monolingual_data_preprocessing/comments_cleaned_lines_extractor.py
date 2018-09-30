import re

class CommentsCleanedTextExtractor:
    NEW_LINE_PREFIX_LOBS = " "
    COMMENT_LEFT_TAG_LOBS = "<"
    COMMENT_RIGHT_TAG_LOBS = ">"
    HASHTAG_COMMENT_SYMBOL = "#"

    def __init__(self, commented_input_file_path: str,
                 comment_left_tag: str,
                 comment_right_tag: str,
                 new_line_prefix: str,
                 ):
        self.commented_input_file_path = commented_input_file_path
        self.comment_left_tag = comment_left_tag
        self.comment_right_tag = comment_right_tag
        self.new_line_prefix = new_line_prefix

    @staticmethod
    def create_cleaned_text_extractor_lobs(commented_cased_input_file_path):
        return CommentsCleanedTextExtractor(commented_cased_input_file_path,
                                            CommentsCleanedTextExtractor.COMMENT_LEFT_TAG_LOBS,
                                            CommentsCleanedTextExtractor.COMMENT_RIGHT_TAG_LOBS,
                                            CommentsCleanedTextExtractor.NEW_LINE_PREFIX_LOBS)

    def is_comment_line_or_empty_line(self, line_stripped: str):
        return (line_stripped.startswith(self.comment_left_tag) and
                line_stripped.endswith(self.comment_right_tag))or \
               line_stripped.startswith(CommentsCleanedTextExtractor.HASHTAG_COMMENT_SYMBOL) \
               or len(line_stripped) <= 0

    def get_non_whitespace_characters(self):
        result = list()

        with open(self.commented_input_file_path, 'r') as file:
                for line in file:

                    line_stripped = line.strip()
                    # Lines that start with a left comment tag
                    # are ignored
                    if not self.is_comment_line_or_empty_line(line_stripped):
                        output_line = ""

                        # The str.split() method without an argument splits on whitespace
                        # https://stackoverflow.com/questions/8113782/split-string-on-whitespace-in-python
                        words = line_stripped.split()
                        for word in words[0:len(words) - 1]:
                            output_line += word
                        # Add the last word
                        word = words[len(words) - 1]
                        output_line += word

                        # https://stackoverflow.com/questions/1276764/
                        # stripping-everything-but-alphanumeric-chars-from-a-string-in-python
                        output_line_alphanumeric_only = re.sub(r'[^a-zA-Z0-9_\-\?\']+', '', output_line)
                        result.extend(list(output_line_alphanumeric_only))
        return result
