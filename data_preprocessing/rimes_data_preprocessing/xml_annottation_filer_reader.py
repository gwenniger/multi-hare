import sys
import xml.etree.ElementTree as ET


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
    def create_rimes_line(rimes_line_element: ET.Element):
        bottom = rimes_line_element.get(BOTTOM_STR)
        right = rimes_line_element.get(RIGHT_STR)
        left = rimes_line_element.get(LEFT_STR)
        top = rimes_line_element.get(TOP_STR)
        line_str = rimes_line_element.get(VALUE_STR)

        bounding_box = BoundingBox(bottom, left, right, top)

        return RimesLine(bounding_box, line_str)

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
        #print(paragraph.attrib)
        rimes_lines = list([])
        paragraph_str = paragraph.get(VALUE_STR)
        for line in XMLAnnotationFileReader.get_lines(paragraph):
            #print(line.attrib)
            rimes_line = RimesLine.create_rimes_line(line)
            #print(rimes_line)
            rimes_lines.append(rimes_line)
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
        paragraphs = list([])
        image_file_name = page.get(FILE_NAME_STR)
        print("document filename: " + str(page.attrib))
        for paragraph in XMLAnnotationFileReader.get_paragraphs(page):
            rimes_paragraph = RimesParagraph.create_rimes_paragraph(paragraph)
            print(rimes_paragraph)
            paragraphs.append(rimes_paragraph)
        return RimesPage(image_file_name, paragraphs)

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
        for page in XMLAnnotationFileReader.get_pages(root):
            rimes_page = RimesPage.create_rimes_page(page)
            result.append(rimes_page)

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


