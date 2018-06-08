
class BoundingBox():
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @staticmethod
    def create_bounding_box(x: int, y: int, w: int, h: int):
        return BoundingBox(x, y, w, h)

    def __str__(self):
        result = "<BoundingBox>" + "\n"
        result += str(self.x) + "," + str(self.y) + "," + str(self.w) + "," + str(self.h) + "\n"
        result += "</BoundingBox>" + "\n"
        return result


class IamLineInformation():

    def __init__(self, line_id: str, is_ok: bool, gray_level: int, number_of_components: int, bounding_box,
                 words: list):
        self.line_id = line_id
        self.is_ok = is_ok
        self.gray_level = gray_level
        self.number_of_components = number_of_components
        self.bounding_box = bounding_box
        self.words = words

    # Extract line parts (including words and meta information) from lines like
    # a01-000x-04 ok 173 25 397 1458 1647 148 and|he|is|to|be|backed|by|Mr.|Will|Griffiths|,
    # a01-000x-05 ok 173 16 393 1635 1082 155 0M P|for|Manchester|Exchange|.
    # Note that first 8 parts concern meta information, after that follows the line contents.
    # Furthermore the line contents can also contain spaces, which needs to be accounted for
    # during the extraction
    @staticmethod
    def get_line_parts(line_string):
        line_parts = line_string.split(" ")

        if len(line_parts) <  9:
            raise RuntimeError("Error: line is not properly formatted, must have at least 9 parts")

        line_id = line_parts[0]

        is_ok_string = line_parts[1]
        is_ok = is_ok_string == "ok"
        gray_level = int(line_parts[2])
        number_of_components = int(line_parts[3])
        bounding_box = BoundingBox.create_bounding_box(int(line_parts[4]), int(line_parts[5]), int(line_parts[6]),
                                                       int(line_parts[7]))

        words_parts = line_parts[8:]

        # Undo the earlier splitting of the word parts
        word_parts_as_single_string = words_parts[0]
        for i in range(1, len(words_parts)):
            word_parts_as_single_string = word_parts_as_single_string + " " + words_parts[i]

        words = word_parts_as_single_string.split("|")

        return line_id, is_ok, gray_level, number_of_components, bounding_box, words

    @staticmethod
    def create_iam_line_information(line_string):
        line_id, is_ok, gray_level, number_of_components, bounding_box, words = \
            IamLineInformation.get_line_parts(line_string)
        return IamLineInformation(line_id, is_ok, gray_level, number_of_components, bounding_box, words)

    def __str__(self):
        result = "<line_information>\n"
        result += "line_id: " + self.line_id + "\n"
        result += "is_ok: " + str(self.is_ok) + "\n"
        result += "gray_level: " + str(self.gray_level) + "\n"
        result += "number_of_components: " + str(self.number_of_components) + "\n"
        result += str(self.bounding_box) + "\n"
        result += "words: " + str(self.words) + "\n"
        result += "</line_information>\n"

        return result

COMMENT_SYMBOL = "#"


class IamLinesDictionary():

    def __init__(self, ok_lines_dictionary, error_lines_dictionary):
        self.ok_lines_dictionary = ok_lines_dictionary
        self.error_lines_dictionary = error_lines_dictionary

    @staticmethod
    def is_comment(line: str):
        return line.startswith(COMMENT_SYMBOL)

    @staticmethod
    def create_iam_dictionary(lines_file_path):

        ok_lines_dictionary = dict([])
        error_lines_dictionary = dict([])

        with open(lines_file_path, "r") as ifile:
            for line in ifile:
                if not IamLinesDictionary.is_comment(line):
                    line_information = IamLineInformation.create_iam_line_information(line)

                    if line_information.is_ok:
                        ok_lines_dictionary[line_information.line_id] = line_information
                    else:
                        error_lines_dictionary[line_information.line_id] = line_information

        return IamLinesDictionary()


def main():
    line1 = "a01-000x-04 ok 173 25 397 1458 1647 148 and|he|is|to|be|backed|by|Mr.|Will|Griffiths|,"
    line2 = "a01-000x-05 ok 173 16 393 1635 1082 155 0M P|for|Manchester|Exchange|."

    line_one_information = IamLineInformation.create_iam_line_information(line1)
    line_two_information = IamLineInformation.create_iam_line_information(line2)

    print("line one information :\n" + str(line_one_information))
    print("line two information :\n" + str(line_two_information))


if __name__ == "__main__":
    main()
