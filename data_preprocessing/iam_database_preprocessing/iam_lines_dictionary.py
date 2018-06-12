
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

    def __eq__(self, other):
        if isinstance(other, BoundingBox):
            return (self.x == other.x) and (self.y == other.y) and (self.w == other.w) and (self.h == other.h)


class IamLineInformation():

    def __init__(self, line_id: str, ok: bool, gray_level: int, number_of_components: int, bounding_box,
                 words: list):
        self.line_id = line_id
        self.ok = ok
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

        ok_string = line_parts[1]
        ok = ok_string == "ok"
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

        return line_id, ok, gray_level, number_of_components, bounding_box, words

    @staticmethod
    def create_iam_line_information(line_string):
        line_id, ok, gray_level, number_of_components, bounding_box, words = \
            IamLineInformation.get_line_parts(line_string)
        return IamLineInformation(line_id, ok, gray_level, number_of_components, bounding_box, words)

    def __str__(self):
        result = "<line_information>\n"
        result += "line_id: " + self.line_id + "\n"
        result += "ok: " + str(self.ok) + "\n"
        result += "gray_level: " + str(self.gray_level) + "\n"
        result += "number_of_components: " + str(self.number_of_components) + "\n"
        result += str(self.bounding_box) + "\n"
        result += "words: " + str(self.words) + "\n"
        result += "</line_information>\n"

        return result

    def is_ok(self):
        return self.ok

    def __eq__(self, other):
        if isinstance(other, IamLineInformation):
            return (self.line_id == other.line_id) and (self.ok == other.ok) \
                   and (self.gray_level == other.gray_level) and \
                   (self.number_of_components == other.number_of_components) and \
                   (self.bounding_box == other.bounding_box) and \
                   (self.words == other.words)

    def get_characters(self):
        result = list([])

        for word in self.words:
            for letter in word:
                result.append(letter)
        return result


class IamLinesDictionary():

    COMMENT_SYMBOL = "#"
    FILE_PATH_SPLIT_SYMBOL = "-"
    PNG_EXTENSION = ".png"

    def __init__(self, ok_lines_dictionary, error_lines_dictionary,
                 iam_database_line_images_root_folder_path: str):
        self.ok_lines_dictionary = ok_lines_dictionary
        self.error_lines_dictionary = error_lines_dictionary
        self.iam_database_line_images_root_folder_path = iam_database_line_images_root_folder_path

    @staticmethod
    def is_comment(line: str):
        return line.startswith(IamLinesDictionary.COMMENT_SYMBOL)

    @staticmethod
    def create_iam_dictionary(lines_file_path: str, iam_database_line_images_root_folder_path: str):

        ok_lines_dictionary = dict([])
        error_lines_dictionary = dict([])

        with open(lines_file_path, "r") as ifile:
            for line in ifile:
                if not IamLinesDictionary.is_comment(line):
                    line_information = IamLineInformation.create_iam_line_information(line)

                    if line_information.ok:
                        ok_lines_dictionary[line_information.line_id] = line_information
                    else:
                        error_lines_dictionary[line_information.line_id] = line_information

        return IamLinesDictionary(ok_lines_dictionary, error_lines_dictionary,
                                  iam_database_line_images_root_folder_path)

    def get_ok_examples(self):
        return self.ok_lines_dictionary.items()

    def get_err_examples(self):
        return self.error_lines_dictionary.items()

    def get_all_examples(self):
        result = dict([])
        result.update(self.ok_lines_dictionary)
        result.update(self.error_lines_dictionary)
        return result

    def get_image_file_path(self, iam_line_information: IamLineInformation):
        line_id = iam_line_information.line_id
        file_path_parts = line_id.split(IamLinesDictionary.FILE_PATH_SPLIT_SYMBOL)

        file_path = self.iam_database_line_images_root_folder_path

        # The somewhat verbose convention to use folder names that are cumulative, that is repeat the
        # parent folder names, is used in the iam database
        cumulative_name = ""
        for i in range(0, len(file_path_parts) - 1):

            if len(cumulative_name) > 0:
                cumulative_name += IamLinesDictionary.FILE_PATH_SPLIT_SYMBOL
            cumulative_name += file_path_parts[i]

            file_path += "/" + cumulative_name

        file_path += "/"

        file_name = line_id + IamLinesDictionary.PNG_EXTENSION
        result = file_path + file_name

        return result


def create_test_iam_line_information_one():
    line_id = "a01-000x-04"
    is_ok = True
    gray_level = 173
    number_of_components = 25
    bounding_box = BoundingBox.create_bounding_box(397, 1458, 1647, 148)
    words = list(["and", "he", "is" , "to" , "be", "backed" , "by" , "Mr.", "Will", "Griffiths", ","])
    return IamLineInformation(line_id, is_ok, gray_level, number_of_components, bounding_box, words)


def test_iam_line_information():
    line1 = "a01-000x-04 ok 173 25 397 1458 1647 148 and|he|is|to|be|backed|by|Mr.|Will|Griffiths|,"
    line2 = "a01-000x-05 ok 173 16 393 1635 1082 155 0M P|for|Manchester|Exchange|."

    line_one_information = IamLineInformation.create_iam_line_information(line1)
    line_two_information = IamLineInformation.create_iam_line_information(line2)

    print("line one information :\n" + str(line_one_information))
    print("line two information :\n" + str(line_two_information))

    if not line_one_information.line_id == "a01-000x-04":
        raise RuntimeError("Error: expected line_id == " + "a01-000x-04")
    
    if not line_one_information.is_ok() is True:
        raise RuntimeError("Error: expected is_ok() == " + str(True))

    reference_bounding_box = BoundingBox.create_bounding_box(397, 1458, 1647, 148)
    if not line_one_information.bounding_box == reference_bounding_box:
        raise RuntimeError("Error: expected " + str(line_one_information.bounding_box) +
                           " == " + str(reference_bounding_box))

    reference_line_information = create_test_iam_line_information_one()
    if not line_one_information == reference_line_information:
        raise RuntimeError("Error: expected equivalence to reference_line_information")


def test_iam_lines_dictionary():
    lines_file_path = "/datastore/data/iam-database/ascii/lines.txt"
    iam_database_line_images_root_folder_path = "/datastore/data/iam-database/lines"
    iam_lines_dicionary = IamLinesDictionary.create_iam_dictionary(lines_file_path,
                                                                   iam_database_line_images_root_folder_path)

    reference_line_information_one = create_test_iam_line_information_one()
    if not(reference_line_information_one.line_id in iam_lines_dicionary.ok_lines_dictionary):
        raise RuntimeError("Error: expected ok_lines_dictionary to contain: " +
                           str(reference_line_information_one))

    for line_information_key in iam_lines_dicionary.ok_lines_dictionary:
        # print("line_information_key: " + str(line_information_key))
        line_information = iam_lines_dicionary.ok_lines_dictionary[line_information_key]
        image_file_path = iam_lines_dicionary.get_image_file_path(line_information)
        # print("image_file_path: " + str(image_file_path))

    return


def main():
    test_iam_line_information()
    test_iam_lines_dictionary()


if __name__ == "__main__":
    main()
