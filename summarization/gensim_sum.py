from gensim.summarization import summarize

from text_processing import preprocessing as preprocess
from utilities.read_write import print_summary_to_file

IDENTIFIER = '_B'


def get_summary(text):
    text = preprocess.clean_and_format(text)
    return summarize(text, ratio=0.1)


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                          'the file you want to summarize: \n')
    OUTPUT_DIR = input('Where do you want it?')

    print_summary_to_file(get_summary, FILE_PATH, OUTPUT_DIR, IDENTIFIER)
