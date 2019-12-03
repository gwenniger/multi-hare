from util.linux_interactor import LinuxInteractor
import sys

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class KenlmInterface:
    KEN_LM_ROOT_DIR_SUFFIX = "libraries/ctcdecode/third_party/kenlm"
    BIN_SUFFIX = "/build/bin/"
    BUILD_ARPA_LANGUAGE_MODEL_PROGRAM_NAME = "lmplz"
    BUILD_BINARY_LANGUAGE_MODEL_PROGRAM_NAME = "build_binary"

    def __init__(self, kenlm_root_dir: str):
        self.kenlm_root_dir = kenlm_root_dir

    @staticmethod
    def create_kenlm_interface(handwriting_recognition_root_dir: str):
        kenlm_root_dir  = handwriting_recognition_root_dir + KenlmInterface.KEN_LM_ROOT_DIR_SUFFIX
        return KenlmInterface(kenlm_root_dir)

    def kenlm_bin_dir(self):
        return self.kenlm_root_dir + KenlmInterface.BIN_SUFFIX

    def kenlm_build_language_model_command_arguments(self, ngram_order: int):
        #return list(["-o", str(ngram_order), "-S", "20%", "--discount_fallback"])
        return list(["-o", str(ngram_order), "-S", "20%"])

    def kenlm_build_arpa_language_model_command(self, ngram_order: int):
        program_arguments = self.kenlm_build_language_model_command_arguments(ngram_order)
        result = list([self.kenlm_bin_dir() + KenlmInterface.BUILD_ARPA_LANGUAGE_MODEL_PROGRAM_NAME])
        result.extend(program_arguments)
        return result

    def kenlm_build_binar_language_model_command(self):
        result = list([self.kenlm_bin_dir() + KenlmInterface.BUILD_BINARY_LANGUAGE_MODEL_PROGRAM_NAME])
        return result

    def create_arpa_language_model_for_file(
            self, ngram_order: int, input_file_path: str, output_file_path: str):
        print("output_file_path: " + str(output_file_path))
        LinuxInteractor.execute_external_command_and_show_output(
            self.kenlm_build_arpa_language_model_command(ngram_order),
            self.kenlm_bin_dir(),
            output_file_path,
            input_file_path)

    def build_binary_language_model_for_file(self, arpa_language_model_file_path, binary_output_file_path):
        command = self.kenlm_build_binar_language_model_command()
        command.append(arpa_language_model_file_path)
        command.append(binary_output_file_path)
        print("Build binary language model command: " + str(command))
        LinuxInteractor.execute_external_command_and_show_output(
            command,
            self.kenlm_bin_dir(),
            None,
            None)


def get_arpa_output_file_path_from_prefix(language_model_file_prefix):
    return language_model_file_prefix + ".arpa"


def get_binary_output_file_path_from_prefix(language_model_file_prefix):
    return language_model_file_prefix + ".binary"


def main():

    if len(sys.argv) != 5:
        for argument in sys.argv[1:]:
            print("program argument: " + argument)

        raise RuntimeError("Error - usage: "
                           "kenlm_build_language_model HANDWRITING_RECOGNITION_ROOT_DIR "
                           "LANGUAGE_MODEL_INPUT_FILE_PATH OUTPUT_FILE_PATH_PREFIX NGRAM_ORDER")

    handwriting_recognition_root_dir = sys.argv[1]
    input_file_path = sys.argv[2]
    output_file_path_prefix = sys.argv[3]
    ngram_order = int(sys.argv[4])

    kenlm_interface = KenlmInterface.create_kenlm_interface(handwriting_recognition_root_dir)
    arpa_file_path = get_arpa_output_file_path_from_prefix(
                                                            output_file_path_prefix
                                                        )
    kenlm_interface.create_arpa_language_model_for_file(ngram_order, input_file_path,
                                                        arpa_file_path
                                                        )
    binary_file_path = get_binary_output_file_path_from_prefix(
        output_file_path_prefix
    )
    kenlm_interface.build_binary_language_model_for_file(arpa_file_path, binary_file_path)


if __name__ == "__main__":
    main()
