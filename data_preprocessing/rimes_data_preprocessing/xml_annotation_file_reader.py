import sys
import xml.etree.ElementTree as ET
import re

"""
The Rimes data set comes with an XML file that describes the data. 
This XML file is a hierarchy of:
 
DocumentList
    SinglePage
        Paragraph
            Line

The line information, which is needed for extracting line strips, must 
be combined with the image file names which is stored at the level of 
SinglePage. To be able to use the information from the XML file in 
a structured way, in this file we provide a set of classes that replicates 
the structure of the XML files in a way that makes the information ready 
to use.
 
"""

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"

BOTTOM_STR = "Bottom"
LEFT_STR = "Left"
RIGHT_STR = "Right"
TOP_STR = "Top"
VALUE_STR = "Value"
FILE_NAME_STR = "FileName"

class BoundingBox:

    def __init__(self, bottom: int, left: int, right: int, top: int):
        self.bottom = bottom
        self.left = left
        self.right = right
        self.top = top


    def __str__(self):
        result = "<BoundingBox>"
        result += "bottom:" + str(self.bottom) + ","
        result += "left:" + str(self.left) + ","
        result += "right:" + str(self.right) + ","
        result += "top:" + str(self.top) + ","
        result += "</BoundingBox>"
        return result


class RimesLine:

    def __init__(self, bounding_box: BoundingBox, line_str: str):
        self.bounding_box = bounding_box
        self.line_str = line_str

    @staticmethod
    def corrected_bounding_box_value(bounding_box_value_string: str):
        # Bounding box values are corrected by replacing negative values with 0
        return max(0, int(bounding_box_value_string))

    @staticmethod
    def create_rimes_line_with_specified_line_string(rimes_line_element: ET.Element, line_str: str):
        bottom = RimesLine.corrected_bounding_box_value(
            rimes_line_element.get(BOTTOM_STR))
        right = RimesLine.corrected_bounding_box_value(
            rimes_line_element.get(RIGHT_STR))
        left = RimesLine.corrected_bounding_box_value(
            rimes_line_element.get(LEFT_STR))
        top = RimesLine.corrected_bounding_box_value(
            rimes_line_element.get(TOP_STR))

        bounding_box = BoundingBox(bottom, left, right, top)

        return RimesLine(bounding_box, line_str)

    @staticmethod
    def create_rimes_line(rimes_line_element: ET.Element):

        line_str = rimes_line_element.get(VALUE_STR)

        return RimesLine.create_rimes_line_with_specified_line_string(rimes_line_element, line_str)

    def __str__(self):
        result = "<RIMES LINE>" + " "
        result += str(self.bounding_box) + " "
        result += "line string: \"" + str(self.line_str) + "\" "
        result += "</RIMES LINE>"
        return result


class RimesParagraph:

    def __init__(self, paragraph_string: str, lines: list):
        self.paragraph_string = paragraph_string
        self.lines = lines

    @staticmethod
    def create_rimes_paragraph(paragraph):
        """
        This function creates a Rimes paragraph, but skips those paragraphs
        that have an inconsistent number of paragraph parts and line items.
        For some reason the rimes database is not entirely clean, and some
        paragraphs apparently have errors in the annotation, so we skip them.
        :param paragraph:
        :return:
        """
        #print(paragraph.attrib)
        rimes_lines = list([])
        paragraph_str = paragraph.get(VALUE_STR).strip()
        # Remove the last newline if any
        if paragraph_str[len(paragraph_str) -2 : len(paragraph_str)] == "\\n":
            paragraph_str = paragraph_str[0:len(paragraph_str) - 2]

        print("paragraph_str: " + paragraph_str)
        # https://stackoverflow.com/questions/6478845/python-split-consecutive-delimiters
        paragraph_parts = re.split("(\\\\n)+", paragraph_str)
        # But the capturing group (\\\\n) remains in the result, see e.g.
        # http://programmaticallyspeaking.com/split-on-separator-but-keep-the-separator-in-python.html
        # So we have to remove it
        # See: https://stackoverflow.com/questions/39919586/how-do-i-ignore-the-group-in-a-regex-split-in-python
        paragraph_parts = paragraph_parts[0::2]

        for i in range(0, len(paragraph_parts)):
            print("paragraph_parts[" + str(i) + "]: " + str(
                paragraph_parts[i]))

        line_index = 0
        lines = XMLAnnotationFileReader.get_lines(paragraph)



        # If the number of lines in the paragraph does not match up with the number
        # of XML line elements, we skip the entire paragraph
        if len(lines) != len(paragraph_parts):
            print(">>> Skipping paragraph because #lines (=" + str(len(lines))
                  + ") != #paragraph_parts (=" + str(len(paragraph_parts)) +") paragraph_str:\n" + paragraph_str)
            print("<lines: >")
            for line in lines:
                print(str(line.get(VALUE_STR)))
            print("</lines>")
            rimes_lines = list([])
        else:
            for line in lines:
                line_str = paragraph_parts[line_index]
                #print(line.attrib)
                # Use the line extracted from the paragraph string. Because the line information in the
                # line parts of the XML file is buggy, possibly because of not dealing well with
                # multiple consecutive \n symbols
                rimes_line = RimesLine.create_rimes_line_with_specified_line_string(line, line_str)
                #rimes_line = RimesLine.create_rimes_line(line)
                #print(rimes_line)
                rimes_lines.append(rimes_line)
                line_index += 1

        return RimesParagraph(paragraph_str, rimes_lines)

    def __str__(self):
        result = "<RimesParagraph>" + "\n"
        result += "pragrah str: " + self.paragraph_string + "\n"
        result += "lines: "  + "\n"
        for line in self.lines:
            result += "\t" +  str(line) + "\n"
        result += "</RimesParagraph>"

        return result

class RimesPage:
    """
    This class stores the Rimes paragraphs associated with a Rimes page
    in a structure way. Importantly this class also stores the image
    file name, needed for retrieving the image.
    In practice for the purpose of extracting line strip images this
    class is the entry point. The get_rimes_lines method retrieves a list
    of RimesLine object from the paragraphs and these objects contain all
    the bounding-box information that in combination with the image
    file is sufficient for extracting the line strips.

    """

    def __init__(self, image_file_name:str, paragraphs: list):
        self.image_file_name = image_file_name
        self.paragraphs = paragraphs

    @staticmethod
    def create_rimes_page(page):
        skipped_paragraphs = 0
        paragraphs = list([])
        image_file_name = page.get(FILE_NAME_STR)
        print("document filename: " + str(page.attrib))
        for paragraph in XMLAnnotationFileReader.get_paragraphs(page):
            rimes_paragraph = RimesParagraph.create_rimes_paragraph(paragraph)
            if len(rimes_paragraph.lines) == 0:
                skipped_paragraphs += 1
            print(rimes_paragraph)
            paragraphs.append(rimes_paragraph)
        return RimesPage(image_file_name, paragraphs), skipped_paragraphs

    def __str__(self):
        result =  "<RimesPage>" + "\n"
        result += "image file name: " + str(self.image_file_name) + "\n"
        for paragraph in self.paragraphs:
            result += str(paragraph) + "\n"
        result += "</RimesPage>"
        return result

    def get_rimes_lines(self):
        result = list([])
        for paragraph in self.paragraphs:
            result.extend(paragraph.lines)
        return result

class XMLAnnotationFileReader:

    def __init__(self, annotation_file_path):
        self.annotation_file_path = annotation_file_path


    @staticmethod
    def get_sub_elements(node):
        result = list([])
        for elem in node:
            result.append(elem)
        return result

    @staticmethod
    def get_pages(root):
        return XMLAnnotationFileReader.get_sub_elements(root)

    @staticmethod
    def get_paragraphs(page):
        return XMLAnnotationFileReader.get_sub_elements(page)

    @staticmethod
    def get_lines(paragraph):
        return XMLAnnotationFileReader.get_sub_elements(paragraph)


    def extract_rimes_pages(self):
        """
        This function extracts a list of RimesParagraph objects from
        the annotation file
        :return:
        """
        print("Extract RIMES paragraphs...")
        tree = ET.parse(self.annotation_file_path)
        root = tree.getroot()

        result = list([])
        # all item attributes
        print('\nAll attributes:')
        skipped_paragraphs = 0
        for page in XMLAnnotationFileReader.get_pages(root):
            rimes_page, skipped_paragraphs_page = RimesPage.create_rimes_page(page)
            skipped_paragraphs += skipped_paragraphs_page
            result.append(rimes_page)

        print("Skipped a total of " + str(skipped_paragraphs) + " paragraphs")

        return result





def main():
    if len(sys.argv) != 2:
        raise RuntimeError(
            "Error: usage: xml_annotation_file_reader XML_INPUT_FILE_PATH")

    input_file_path = sys.argv[1]
    print("input_file_path: " + str(input_file_path))
    xml_annotation_file_reader = XMLAnnotationFileReader(input_file_path)
    rimes_pages = xml_annotation_file_reader.extract_rimes_pages()

    for rimes_page in rimes_pages:
        print("rimes page: \n\n" + str(rimes_page))


if __name__ == "__main__":
    main()


