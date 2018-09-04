from util.linux_interactor import LinuxInteractor


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

