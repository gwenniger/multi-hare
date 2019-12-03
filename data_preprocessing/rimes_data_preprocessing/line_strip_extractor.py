import sys
import cv2

from data_preprocessing.rimes_data_preprocessing.xml_annotation_file_reader import XMLAnnotationFileReader
from data_preprocessing.rimes_data_preprocessing.xml_annotation_file_reader import RimesLine

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"

LINE_STRIPS_OUTPUT_FOLDER_NAME = "line_strips"
LINE_STRIPS_IMPROVED_OUTPUT_FOLDER_NAME = "line_strips_improved"

class LineStripExtractor:

    def __init__(self, rimes_pages: list, data_root_folder: str):
        self.rimes_pages = rimes_pages
        self.data_root_folder = data_root_folder


    @staticmethod
    def create_line_strip_extractor(annotation_file_path: str,
                                    data_root_folder: str):
        xml_annotation_file_reader = XMLAnnotationFileReader(annotation_file_path)
        rimes_pages = xml_annotation_file_reader.extract_rimes_pages()
        return LineStripExtractor(rimes_pages, data_root_folder)


    @staticmethod
    def line_strips_output_folder_static(data_root_folder):
        return data_root_folder + LINE_STRIPS_OUTPUT_FOLDER_NAME + "/"

    @staticmethod
    def line_strips_improved_output_folder_static(data_root_folder):
        return data_root_folder + LINE_STRIPS_IMPROVED_OUTPUT_FOLDER_NAME + "/"

    @staticmethod
    def line_strip_image_output_file_name_static(data_root_folder: str,
                                                 example_number: int):
        output_name = LineStripExtractor.\
                          line_strips_output_folder_static(data_root_folder) + \
                      "example_" + str(example_number) + ".png"
        return output_name

    @staticmethod
    def line_strip_image_improved_output_file_name_static(data_root_folder: str,
                                                 example_number: int):
        output_name = LineStripExtractor. \
                          line_strips_improved_output_folder_static(data_root_folder) + \
                      "example_" + str(example_number) + ".png"
        return output_name

    def line_strip_image_output_file_name(self, example_number: int):
        return LineStripExtractor.line_strip_image_output_file_name_static(
            self.data_root_folder, example_number)

    def line_strip_image_improved_output_file_name(self, example_number: int):
        return LineStripExtractor.line_strip_image_improved_output_file_name_static(
            self.data_root_folder, example_number)

    def line_strips_output_folder(self):
        return LineStripExtractor.line_strips_output_folder_static(
            self.data_root_folder)

    def extract_line_strip(self, image_file_path, rimes_line: RimesLine, example_number: int):
        full_image_file_path = self.data_root_folder + image_file_path
        print("image file path: " + str(full_image_file_path))
        # By default images are read with three color channels, to work with
        # grayscale images this must be specified
        # See: https://stackoverflow.com/questions/18870603/in-opencv-python-why-am-i-getting-3-channel-images-from-a-grayscale-image
        image = cv2.imread(full_image_file_path, cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('image', image)
        #cv2.waitKey(0)
        min_x = rimes_line.bounding_box.left
        max_x = rimes_line.bounding_box.right
        max_y = rimes_line.bounding_box.bottom
        min_y = rimes_line.bounding_box.top
        region_of_interest = image[min_y:max_y, min_x:max_x]
        print("miny: " + str(min_y) + " max_y: " + str(max_y))

        output_name = self.line_strip_image_output_file_name(example_number)
        print("writing line strip to file: " + output_name)
        cv2.imwrite(output_name, region_of_interest)



    def extract_line_strips(self):
        example_number = 1
        for rimes_page in self.rimes_pages:
            print("image file path: " + str(rimes_page.image_file_name))
            rimes_lines = rimes_page.get_rimes_lines()
            for rimes_line in rimes_lines:
                print(str(rimes_line))
                self.extract_line_strip(rimes_page.image_file_name,
                                        rimes_line, example_number)
                example_number += 1
        print("Finished - extracted a total of " + str(example_number) +
              " line strips")


def main():
    if len(sys.argv) != 3:
        raise RuntimeError(
            "Error: usage: line_strip_extractor XML_INPUT_FILE_PATH LINE_STRIPS_OUTPUT_FOLDER")

    input_file_path = sys.argv[1]
    data_root_folder = sys.argv[2]

    print("Hello world")
    line_strip_extractor = LineStripExtractor.create_line_strip_extractor(input_file_path, data_root_folder)
    line_strip_extractor.extract_line_strips()


if __name__ == "__main__":
    main()

