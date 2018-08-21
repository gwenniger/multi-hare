from skimage import io
from collections import OrderedDict


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


class IamDataPointInformation():
    def __init__(self, line_id: str, ok: bool, gray_level: int, bounding_box):
        self.line_id = line_id
        self.ok = ok
        self.gray_level = gray_level
        self.bounding_box = bounding_box

    def is_ok(self):
        return self.ok


class IamWordInformation(IamDataPointInformation):

    def __init__(self, line_id: str, ok: bool, gray_level: int, bounding_box,
                 tag: str, word: str):

        super(IamWordInformation, self).__init__(line_id, ok, gray_level, bounding_box)

        self.tag = tag
        self.word = word

    # Extract line parts (including words and meta information) from lines like
    # a01-000u-00-00 ok 154 408 768 27 51 AT A
    # a01-000u-00-01 ok 154 507 766 213 48 NN MOVE
    # Note that first 8 parts concern meta information, after that follows the line contents.
    # Furthermore the line contents can also contain spaces, which needs to be accounted for
    # during the extraction
    @staticmethod
    def get_line_parts(line_string):
        line_parts = line_string.split(" ")

        if len(line_parts) < 9:
            raise RuntimeError("Error: line is not properly formatted, must have at least 9 parts")

        line_id = line_parts[0]

        ok_string = line_parts[1]
        ok = ok_string == "ok"
        gray_level = int(line_parts[2])
        bounding_box = BoundingBox.create_bounding_box(int(line_parts[3]), int(line_parts[4]), int(line_parts[5]),
                                                       int(line_parts[6]))

        tag = line_parts[7]
        word = line_parts[8]

        return line_id, ok, gray_level, bounding_box, tag, word

    @staticmethod
    def create_iam_word_information(line_string):
        line_id, ok, gray_level, bounding_box, tag, word = \
            IamWordInformation.get_line_parts(line_string)
        return IamWordInformation(line_id, ok, gray_level, bounding_box, tag, word)

    def __str__(self):
        result = "<word_information>\n"
        result += "line_id: " + self.line_id + "\n"
        result += "ok: " + str(self.ok) + "\n"
        result += "gray_level: " + str(self.gray_level) + "\n"
        result += str(self.bounding_box) + "\n"
        result += "tag: " + str(self.tag) + "\n"
        result += "word: " + str(self.word) + "\n"
        result += "</word_information>\n"

        return result

    def get_characters(self):
        result = list([])

        for letter in self.word:
            result.append(letter)
        return result


class IamLineInformation(IamDataPointInformation):
    WORD_SEPARATOR_SYMBOL = "|"

    def __init__(self, line_id: str, ok: bool, gray_level: int, bounding_box,
                number_of_components: int, words: list):

        super(IamLineInformation, self).__init__(line_id, ok, gray_level, bounding_box)
        self.number_of_components = number_of_components
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

        if len(line_parts) < 9:
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

        words = word_parts_as_single_string.split(IamLineInformation.WORD_SEPARATOR_SYMBOL)

        return line_id, ok, gray_level, number_of_components, bounding_box, words

    @staticmethod
    def create_iam_line_information(line_string):
        line_id, ok, gray_level, number_of_components, bounding_box, words = \
            IamLineInformation.get_line_parts(line_string)
        return IamLineInformation(line_id, ok, gray_level,bounding_box, number_of_components, words)

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

    def __eq__(self, other):
        if isinstance(other, IamLineInformation):
            return (self.line_id == other.line_id) and (self.ok == other.ok) \
                   and (self.gray_level == other.gray_level) and \
                   (self.number_of_components == other.number_of_components) and \
                   (self.bounding_box == other.bounding_box) and \
                   (self.words == other.words)

    def get_characters(self):
        result = list([])

        last_word_index = len(self.words)-1
        for word in self.words[0:last_word_index]:
            for letter in word:
                result.append(letter)
            # Add a word separator symbol
            result.append(IamLineInformation.WORD_SEPARATOR_SYMBOL)
        # Add the letters of the last word
        for letter in self.words[last_word_index]:
            result.append(letter)
        return result


class IamExamplesDictionary():

    COMMENT_SYMBOL = "#"
    FILE_PATH_SPLIT_SYMBOL = "-"
    PNG_EXTENSION = ".png"
    # Minimum dimensions used for rejecting probably erroneous examples
    MIN_WIDTH_REQUIRED = 8
    MIN_HEIGHT_REQUIRED = 8

    def __init__(self, ok_lines_dictionary, error_lines_dictionary,
                 size_rejected_images_lines_dictionary,
                 iam_database_line_images_root_folder_path: str,
                 get_file_path_part_function):
        self.ok_lines_dictionary = ok_lines_dictionary
        self.error_lines_dictionary = error_lines_dictionary
        self.size_rejected_images_lines_dictionary = size_rejected_images_lines_dictionary
        self.iam_database_line_images_root_folder_path = iam_database_line_images_root_folder_path
        self.get_file_path_part_function = get_file_path_part_function

    @staticmethod
    def is_comment(line: str):
        return line.startswith(IamExamplesDictionary.COMMENT_SYMBOL)

    # Not all image files are actually proper, some don't show a proper
    # image for the line/word. Therefore minimum dimensions are used to reject overly small
    # images
    @staticmethod
    def image_has_minimal_dimensions(line_information, iam_database_line_images_root_folder_path: str,
                                     get_file_path_part_function):
        image_file_path = IamExamplesDictionary. \
            get_image_file_path_static(line_information,
                                       iam_database_line_images_root_folder_path,
                                       get_file_path_part_function)
        image = io.imread(image_file_path)
        # print(">>> image_has_minimal_dimensions - image.shape: " + str(image.shape))
        height, width = image.shape
        if height < IamExamplesDictionary.MIN_HEIGHT_REQUIRED or width < IamExamplesDictionary.MIN_WIDTH_REQUIRED:
            print("Rejecting image " + image_file_path + " of size " + str(image.shape) +
                  " since it is not satisfying the minimum height(" + str(IamExamplesDictionary.MIN_HEIGHT_REQUIRED) +
                  " and minimum width (" + str(IamExamplesDictionary.MIN_HEIGHT_REQUIRED) +
                  " requirements")
            return False
        return True

    @staticmethod
    def create_iam_dictionary(lines_file_path: str, iam_database_line_images_root_folder_path: str,
                              information_creation_function,
                              require_min_image_size: bool, get_file_path_part_function):

        # Ordered dictionaries should be used so that enumerations over the
        # items added to the dictionaries have a deterministic order
        # corresponding to the order in which items were added
        ok_lines_dictionary = OrderedDict([])
        error_lines_dictionary = OrderedDict([])
        size_rejected_images_lines_dictionary = OrderedDict([])

        total_ok_images = 0
        number_of_rejected_images_labeled_ok = 0
        with open(lines_file_path, "r") as ifile:
            for line in ifile:
                # The lines end with a new line character and hence need
                # to be stripped to get rid of those otherwise disrupting newline
                # characters
                # See: https://stackoverflow.com/questions/12330522/reading-a-file-without-newlines
                line = line.rstrip('\n')
                if not IamExamplesDictionary.is_comment(line):
                    line_information = information_creation_function(line)

                    if line_information.ok:

                        image_is_acceptable = True

                        if require_min_image_size:
                            image_is_acceptable = IamExamplesDictionary.image_has_minimal_dimensions(
                                line_information, iam_database_line_images_root_folder_path, get_file_path_part_function)

                        if not image_is_acceptable:
                            size_rejected_images_lines_dictionary[line_information.line_id] = line_information
                            number_of_rejected_images_labeled_ok += 1
                        else:
                            ok_lines_dictionary[line_information.line_id] = line_information
                        total_ok_images += 1
                    else:
                        error_lines_dictionary[line_information.line_id] = line_information

            print("Rejected in total " + str(number_of_rejected_images_labeled_ok) + " of the " +
                  str(total_ok_images) + " ok labeled images, since they do not satisfy " +
                  "the minimum size requirements")

        return IamExamplesDictionary(ok_lines_dictionary, error_lines_dictionary,
                                     size_rejected_images_lines_dictionary,
                                     iam_database_line_images_root_folder_path,
                                     get_file_path_part_function)

    @staticmethod
    def create_iam_lines_dictionary(lines_file_path: str, iam_database_line_images_root_folder_path: str,
                                    require_min_image_size: bool
                                    ):
        return IamExamplesDictionary.create_iam_dictionary(lines_file_path, iam_database_line_images_root_folder_path,
                                                           IamLineInformation.create_iam_line_information,
                                                           require_min_image_size,
                                                           IamExamplesDictionary.get_file_sub_path_for_line)

    @staticmethod
    def create_iam_words_dictionary(lines_file_path: str, iam_database_line_images_root_folder_path: str,
                                    require_min_image_size: bool
                                    ):
        return IamExamplesDictionary.create_iam_dictionary(lines_file_path, iam_database_line_images_root_folder_path,
                                                           IamWordInformation.create_iam_word_information,
                                                           require_min_image_size,
                                                           IamExamplesDictionary.get_file_sub_path_for_word)

    def get_ok_examples(self):
        print("get_ok_examples - number of OK examples: "
              + str(len(self.ok_lines_dictionary.items())))
        return self.ok_lines_dictionary.items()

    def get_err_examples(self):
        return self.error_lines_dictionary.items()

    def get_all_examples(self):
        result = OrderedDict([])
        result.update(self.ok_lines_dictionary)
        result.update(self.error_lines_dictionary)
        return result

    @staticmethod
    def get_file_sub_path(file_path, file_path_parts):
        # print(">>>file_path_parts: " + str(file_path_parts))

        # The somewhat verbose convention to use folder names that are cumulative, that is repeat the
        # parent folder names, is used in the iam database
        cumulative_name = ""
        for i in range(0, len(file_path_parts) - 1):

            if len(cumulative_name) > 0:
                cumulative_name += IamExamplesDictionary.FILE_PATH_SPLIT_SYMBOL
            cumulative_name += file_path_parts[i]

            file_path += "/" + cumulative_name

        file_path += "/"
        return file_path

    @staticmethod
    def get_file_sub_path_for_line(file_path, line_id):
        file_path_parts = line_id.split(IamExamplesDictionary.FILE_PATH_SPLIT_SYMBOL)

        return IamExamplesDictionary.get_file_sub_path(file_path, file_path_parts)

    @staticmethod
    def get_file_sub_path_for_word(file_path, line_id):
        file_path_parts = line_id.split(IamExamplesDictionary.FILE_PATH_SPLIT_SYMBOL)
        # print(">>>file_path_parts: " + str(file_path_parts))

        return IamExamplesDictionary.get_file_sub_path(file_path,
                                                       file_path_parts[0:len(file_path_parts) -1])

    @staticmethod
    def get_image_file_path_static(iam_line_information: IamLineInformation,
                                   iam_database_line_images_root_folder_path: str,
                                   get_file_path_part_function
                                   ):
        line_id = iam_line_information.line_id
        # print(">>> line_id: " + str(line_id))
        file_path = iam_database_line_images_root_folder_path
        file_path = get_file_path_part_function(file_path, line_id)


        file_name = line_id + IamExamplesDictionary.PNG_EXTENSION
        result = file_path + file_name

        return result

    def get_image_file_path(self, iam_line_information: IamLineInformation):
        return IamExamplesDictionary.get_image_file_path_static(iam_line_information,
                                                                self.iam_database_line_images_root_folder_path,
                                                                self.get_file_path_part_function)


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
    iam_lines_dicionary = IamExamplesDictionary.create_iam_lines_dictionary(lines_file_path,
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
