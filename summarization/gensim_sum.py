import codecs

from gensim.summarization import summarize

from preprocessing import preprocessing as preprocess
from utilities.utils import read_file


def get_summary(text, sentence_num=10):
    text = preprocess.clean_and_format(text)
    return summarize(text)


def print_to_file(lines, filename):
    file = codecs.open(filename, 'w')
    print(lines, file=file)


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                          'the file you want to summarize: \n')
    FILE_TEXT = read_file(FILE_PATH)
    summary = get_summary(FILE_TEXT)

    path = FILE_PATH.split('.')
    path.pop()
    path.append('summ')
    OUTPUT_FILE_PATH = '.'.join(path)

    print_to_file(summary, OUTPUT_FILE_PATH)
