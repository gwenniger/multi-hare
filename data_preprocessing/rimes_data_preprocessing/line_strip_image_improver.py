import subprocess
import sys
from util.file_utils import FileUtils

IMAGE_DENSITY = 118.110

class LineStripImageImprover:
    """
    This class creates improved images using https://github.com/mauvilsa/imgtxtenh
    to improve noisy scanned text images and

    convert (from ImageMagick) to straighten lines.
    This is a re-implementation of the image improvement from
    https://github.com/jpuigcerver/Laia/blob/master/egs/rimes/steps/prepare.sh
    (but without scaling images to a fixed height).

    """

    def __init__(self, imgtxtenh_bin_path: str):
        self.imgtxtenh_bin_path = imgtxtenh_bin_path

    @staticmethod
    def convert_and_deskew_command(output_file_path: str):
        result_string = "convert -colorspace Gray -deskew 40%"
        result_string += " -bordercolor white -border 5 -trim"
        result_string += " -bordercolor white -border 20"
        result_string += "x0 +repage -strip - " +  output_file_path
        print("convert_and_deskew_command_string: " + str(result_string))
        result_string_as_list = result_string.split()
        return result_string_as_list

    def create_improved_image(self, input_image_file_path: str, output_image_file_path: str):
        print("Creating improved image from: " + str(input_image_file_path))
        ps = subprocess.Popen((self.imgtxtenh_bin_path, input_image_file_path, '-d', str(IMAGE_DENSITY)), stdout=subprocess.PIPE)
        # Adapted from https://github.com/jpuigcerver/Laia/blob/master/egs/rimes/steps/prepare.sh
        subprocess.check_output((LineStripImageImprover.convert_and_deskew_command(output_image_file_path)),
                                         stdin=ps.stdout)
        print("Improved image was saved to: " + str(output_image_file_path))
        ps.wait()


    def create_improved_images(self, input_images_folder_path: str, output_images_folder_path: str):
        for input_file_name in FileUtils.get_all_files_in_directory(input_images_folder_path, False):
            input_file_path_complete = input_images_folder_path + "/" + input_file_name
            output_file_path_complete = output_images_folder_path + "/" + input_file_name
            self.create_improved_image(input_file_path_complete, output_file_path_complete)




def main():
    for index, value in enumerate(sys.argv):
        print("sys.argv[" + str(index) + "]: " + str(value))

    if len(sys.argv) != 4:
        raise RuntimeError(
            "Error: usage: line_strip_image_improver IMGTXTENH_BIN_PATH, INPUT_IMAGES_FOLDER OUTPUT_IMAGES_FOLDER ")

    imgtxtenh_bin_path = sys.argv[1]
    input_images_folder_path = sys.argv[2]
    output_images_folder_path = sys.argv[3]

    print("Hello world")
    line_strip_image_improver = LineStripImageImprover(imgtxtenh_bin_path)
    line_strip_image_improver.create_improved_images(input_images_folder_path, output_images_folder_path)

if __name__ == "__main__":
    main()

