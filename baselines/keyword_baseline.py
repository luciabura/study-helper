"""
This is a baseline implementation for extracting the keywords of a given block of text.
The number of keywords can be given as in input n to the baseline function and will be returning
the first n words in the document which have the highest frequency count.

I have chosen this as a baseline after consideration of general use of language and
key concepts when appearing within a scientific paper or teaching material.

"""
from nltk import FreqDist

from preprocessing.preprocessing import word_tokenize
from utilities.utils import read_file


def get_most_common(words, word_count=1):
    """Print on screen the first `word_count` words that have the
    highest frequency count"""
    counts = FreqDist(words)
    return [word for word, _ in counts.most_common(word_count)]


def get_keywords(file_text, word_count=1):
    words = word_tokenize(file_text)
    keywords = get_most_common(words, word_count)
    return keywords


if __name__ == '__main__':
    FILE_PATH = raw_input('Enter the absolute path of '
                          'the file you want to extract the keywords from: \n')
    FILE_TEXT = read_file(FILE_PATH)
    print get_keywords(FILE_TEXT, 4)
