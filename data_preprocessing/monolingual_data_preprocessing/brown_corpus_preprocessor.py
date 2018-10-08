import sys
from data_preprocessing.monolingual_data_preprocessing.lob_original_preprocessor import LobOriginalPreprocessor
from nltk.corpus import brown
import nltk
nltk.download('brown')


class BrownCorpusPreprocessor:

    def __init__(self, output_file_path: str):
        self.output_file_path = output_file_path

    @staticmethod
    def create_brown_corpus_preprocessor(output_file_path: str):
        return BrownCorpusPreprocessor(output_file_path)

    @staticmethod
    def replace_quotation_marks(sentence: str):
        """
        Replace the quotation marks in the brown corpus with the
        type used in the LOB corpus.
        :param sentence:
        :return:
        """
        result = sentence.replace("``", "\"")
        result = result.replace("''", "\"")
        return result

    @staticmethod
    def replace_doubled_punctuation(sentence: str):
        # For some obscure reason these symbols get doubled in the brown corpus,
        # so we need to remove th second one
        result = sentence.replace("? ?", "?")
        result = result.replace("! !", "!")
        result = result.replace("; ;", ";")
        return result

    def write_preprocessed_brown_corpus_to_output_file(self):
        print("writing output to: " + self.output_file_path)
        # https://stackoverflow.com/questions/47301140/how-can-i-access-the-raw-documents-from-the-brown-corpus
        with open(self.output_file_path, "w") as output_file:
            for sent in brown.sents():
                sentence_preprocessed = " ".join(sent)
                sentence_preprocessed = BrownCorpusPreprocessor.replace_quotation_marks(sentence_preprocessed)
                # Do some apostrophe re-tokenization mimic the weird tokenization used in the
                # iam dataset of apostrophes in verbs
                sentence_preprocessed = LobOriginalPreprocessor.\
                    perform_iam_specific_apostrophe_re_tokenization(sentence_preprocessed)
                sentence_preprocessed = BrownCorpusPreprocessor.replace_doubled_punctuation(sentence_preprocessed)
                output_file.write(sentence_preprocessed + "\n")


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Error: brown_corpus_preprocessor OUTPUT_FILE_PATH")

    output_file_path = sys.argv[1]
    brown_corpus_preprocessor = BrownCorpusPreprocessor.create_brown_corpus_preprocessor(output_file_path)
    brown_corpus_preprocessor.write_preprocessed_brown_corpus_to_output_file()


if __name__ == "__main__":
    main()
