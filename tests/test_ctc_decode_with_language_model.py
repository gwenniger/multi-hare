import sys
from language_model.kenlm_interface import KenlmInterface

class TestLanguageModelCreator:
    # HANDWRITING_RECOGNITION_ROOT_DIR = "~/AI/handwriting-recognition/"
    LANGUAGE_MODEL_OUTPUT_DIR_SUFFIX = "tests/language_models/"
    LANGUAGE_MODEL_TRAIN_FILE_NAME = "language_model_train_file.txt"
    LANGUAGE_MODEL_ARPA_FILE_NAME = "test_language_model.arpa"
    LANGUAGE_MODEL_TEXT = "Klaas maakt een keuken met een kraan .\n" \
                          "De kraan die hij maakt is aan de kant van de " \
                          "muur .\nDe kraan is duur , maar niet te huur .\n" \
                          "De huur is niet duur. Maar huren van muren " \
                          "kan ook rare kuren sturen .\n" \
                          "Huur een keuken met een kraan .\n"

    def __init__(self, handwriting_recognition_root_dir: str,
                 language_model_output_directory: str, language_model_train_file_name: str,
                 language_model_arpa_file_name: str):
        self.handwriting_recognition_root_dir = handwriting_recognition_root_dir
        self.language_model_output_directory = language_model_output_directory
        self.language_model_train_file_name = language_model_train_file_name
        self.language_model_arpa_file_name = language_model_arpa_file_name

    @staticmethod
    def create_test_language_model_creator(handwriting_recognition_root_dir):
        language_model_output_directory = handwriting_recognition_root_dir +\
            TestLanguageModelCreator.LANGUAGE_MODEL_OUTPUT_DIR_SUFFIX
        language_model_train_file_name = TestLanguageModelCreator.LANGUAGE_MODEL_TRAIN_FILE_NAME

        return TestLanguageModelCreator(handwriting_recognition_root_dir,
                                        language_model_output_directory, language_model_train_file_name,
                                        TestLanguageModelCreator.LANGUAGE_MODEL_ARPA_FILE_NAME)

    def get_language_model_train_file_path(self):
        return self.language_model_output_directory + self.language_model_train_file_name

    def get_language_model_arpa_file_path(self):
        return self.language_model_output_directory + self.language_model_arpa_file_name

    def create_language_model_train_file(self, language_model_text: str):
        with open(self.get_language_model_train_file_path(), "w") as text_file:
            text_file.write(language_model_text)

    def create_language_model_arpa_file(self, ngram_order: int):
        kenlm_interface = KenlmInterface.create_kenlm_interface(
            self.handwriting_recognition_root_dir)
        kenlm_interface.create_language_model_for_file(
            ngram_order, self.get_language_model_train_file_path(), self.get_language_model_arpa_file_path())


class TestCTCDecodeWithLanguageModel:

    def __init__(self):
        return


def create_test_language_model(handwriting_recognition_root_dir: str):
    test_language_model_creator = TestLanguageModelCreator.create_test_language_model_creator(
        handwriting_recognition_root_dir)
    test_language_model_creator.create_language_model_train_file(TestLanguageModelCreator.LANGUAGE_MODEL_TEXT)
    # For a higher order language model a bigger, non-artificial training set may be required.
    test_language_model_creator.create_language_model_arpa_file(1)


def main():
    handwriting_recognition_root_dir = sys.argv[1]
    create_test_language_model(handwriting_recognition_root_dir)


if __name__ == "__main__":
    main()


