
class RimesLineInformation:
    WORD_SEPARATOR_SYMBOL = "|"

    def __init__(self, image_file_path: str, words: list):

        self.image_file_path = image_file_path
        self.words = words


    @staticmethod
    def get_line_parts(line_string):
        line_parts = line_string.split(" ")

        if len(line_parts) < 2:
            raise RuntimeError("Error: line is not properly formatted, must have at least 2 parts")

        image_file_path = line_parts[0]
        words_parts = line_parts[1:]

        # Undo the earlier splitting of the word parts
        word_parts_as_single_string = words_parts[0]
        for i in range(1, len(words_parts)):
            word_parts_as_single_string = word_parts_as_single_string + " " + words_parts[i]

        words = word_parts_as_single_string.split(RimesLineInformation.WORD_SEPARATOR_SYMBOL)

        return image_file_path, words

    @staticmethod
    def create_rimes_line_information(line_string):
        image_file_path, words = \
            RimesLineInformation.get_line_parts(line_string)
        return RimesLineInformation(image_file_path, words)

    def __str__(self):
        result = "<line_information>\n"
        result += "image_file_path: " + self.image_file_path + "\n"
        result += "words: " + str(self.words) + "\n"
        result += "</line_information>\n"

        return result

    def __eq__(self, other):
        if isinstance(other, RimesLineInformation):
            return (self.image_file_path == other.image_file_path) and \
                   (self.words == other.words)

    """
        This method collects and adds all the characters of the words including the 
        word separator between words. It is not entirely clear whether the word separator 
        should be added or not. From the point of character prediction perhaps not. At the 
        same time, how about ambiguous sequences like "backingate"
    """
    def get_characters_with_word_separator(self):
        result = list([])

        last_word_index = len(self.words)-1
        for word in self.words[0:last_word_index]:
            for letter in word:
                result.append(letter)
            # Add a word separator symbol
            result.append(RimesLineInformation.WORD_SEPARATOR_SYMBOL)
        # Add the letters of the last word
        for letter in self.words[last_word_index]:
            result.append(letter)
        return result

    def get_characters(self):
        result = list([])

        for word in self.words:
            for letter in word:
                result.append(letter)
        return result

    @staticmethod
    def is_ok():
        """
        For compatibility with IAMExamplesDictionary
        """
        return True

    def line_id(self):
        """
        For compatibility with IAMExamplesDictionary
        """
        return self.image_file_path