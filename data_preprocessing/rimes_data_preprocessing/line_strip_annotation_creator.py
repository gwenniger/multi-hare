import sys
from data_preprocessing.rimes_data_preprocessing.xml_annotation_file_reader import XMLAnnotationFileReader
from data_preprocessing.rimes_data_preprocessing.line_strip_extractor import LineStripExtractor
from data_preprocessing.rimes_data_preprocessing.rimes_tokenize import CustomTreebankWordTokenizer

class LineStripAnnotationCreator:

    def __init__(self, rimes_pages: list, line_strips_annotation_file_path: str,
                 line_strips_annotation_improved_file_path: str,
                 data_root_folder: str):
        self.rimes_pages = rimes_pages
        self.line_strips_annotation_file_path = line_strips_annotation_file_path
        self.line_strips_annotation_improved_file_path = line_strips_annotation_improved_file_path
        self.data_root_folder = data_root_folder

    @staticmethod
    def create_line_strip_annotation_creator(annotation_file_path: str,
                                             data_root_folder: str):
        xml_annotation_file_reader = XMLAnnotationFileReader(annotation_file_path)
        rimes_pages = xml_annotation_file_reader.extract_rimes_pages()
        line_strips_annotation_file_path = data_root_folder + "line_strips_annotation.txt"
        line_strips_annotation_improved_file_path = data_root_folder + "line_strips_annotation_improved.txt"
        return LineStripAnnotationCreator(rimes_pages, line_strips_annotation_file_path,
                                          line_strips_annotation_improved_file_path, data_root_folder)


    def create_annotation_file(self, use_improved_images: bool):
        tokenizer = CustomTreebankWordTokenizer()
        if use_improved_images:
            output_file = open(self.line_strips_annotation_improved_file_path,
                               "w")
        else:
            output_file = open(self.line_strips_annotation_file_path, "w")

        example_number = 1
        for rimes_page in self.rimes_pages:
            #print("image file path: " + str(rimes_page.image_file_name))
            rimes_lines = rimes_page.get_rimes_lines()
            for rimes_line in rimes_lines:
                #print(str(rimes_line))
                if use_improved_images:
                    line_strip_image_name = LineStripExtractor.line_strip_image_improved_output_file_name_static(
                        self.data_root_folder, example_number)
                else:
                    line_strip_image_name = LineStripExtractor.line_strip_image_output_file_name_static(
                        self.data_root_folder, example_number)
                output_file.write(line_strip_image_name + " ")
                line_tokens = tokenizer.tokenize(rimes_line.line_str)
                output_file.write('|'.join(map(str, line_tokens)) + "\n")
                # See: https://stackoverflow.com/questions/2399112/python-print-delimited-list

                example_number += 1
        print("Finished - extracted a total of " + str(example_number) +
              " line strips")
        output_file.close()

def main():
    if len(sys.argv) != 3:
        raise RuntimeError(
            "Error: usage: line_strip_annotation_creator XML_INPUT_FILE_PATH ROOT_FOLDER_PATH")

    input_file_path = sys.argv[1]
    data_root_folder = sys.argv[2]

    print("Hello world")
    line_strip_extractor = LineStripAnnotationCreator.create_line_strip_annotation_creator(
        input_file_path, data_root_folder)
    # Create annotation file for normal images
    line_strip_extractor.create_annotation_file(False)
    # Create annotation file for improved images
    line_strip_extractor.create_annotation_file(True)

if __name__ == "__main__":
    main()

