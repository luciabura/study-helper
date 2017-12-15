from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import nltk
import spacy

import sys

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

NLP = spacy.load('en')


def clean_and_tokenize(text):
    """
    :param text: Original unprocessed text (as read directly from file)
    :return: The same text, in the form of a list of tokens given by
    spacy's tokenization
    """
    text = clean_and_format(text)
    tokens = NLP(text)
    # for token in tokens:
    #     print(token.text + ": " + token.tag_)

    return tokens


def clean_and_format(text):
    """
    Strips punctuation and makes sure all text is presented in
    unicode form, with no newlines or multiple spaces
    :param text: the original text, in ascii or unicode
    :return: unicode formatted text
    """

    # ensure no '\n' s in text
    lines = text.splitlines()
    text = ' '.join(lines)

    # ensure not more than 1 space
    words = text.split()
    text = ' '.join(words)

    # get rid of everything except . , ? and '
    text = _strip_punctuation(text, partial=True)
    return text


def remove_stopwords(tokens):
    """
    :param tokens: a list of tokens as given by spacy's tokenization
    (eg: used after clean_and_tokenize)
    :return: a list of tokens which don't contain stopwords (a, such, for etc...)
    """
    without_stopwords = [token for token in tokens if token.text not in STOP_WORDS]
    return without_stopwords


def nltk_sentence_tokenize(text):
    # text = clean_and_format(text)
    return nltk.sent_tokenize(text)

def sentence_tokenize(text):
    text = clean_and_format(text)
    doc = NLP(text)
    sentences = [sent for sent in doc.sents]
    return sentences


def nltk_word_tokenize(text):
    """Perform the preprocessing steps on the text: remove stop words
    and punctuation, convert to lowercase, tokenize"""
    text = _strip_punctuation(text)
    data = nltk.word_tokenize(text)
    data = [word.lower() for word in data]
    text_data = nltk.Text(data)
    words = nltk_remove_stopwords(text_data)
    return words


def pos_tokenize(tokens):
    return nltk.pos_tag(tokens)


def lemmatize_word(word):
    return LEMMATIZER.lemmatize(word)


def remove_hyphens(words):
    """
    :param words: A list of words, some of which may contain hyphens
    :return: The same list, but all hyphenated words are split into parts
    and added back, in the same order
    """
    hyphenated = []

    # will break if I have cross correlation and cross-correlation will always just take second
    for i, word in enumerate(words):
        hyph_parts = word.split('-')
        if len(hyph_parts) > 1:
            hyphenated.append(word)
            words.pop(i)
            for part in reversed(hyph_parts):
                if len(part) > 0:
                    words.insert(i, part)

    return words, hyphenated

"""Internal functions"""
RE_SOME_PUNCT = re.compile('[^\w\'.,?;]+', re.UNICODE)
RE_ALL_PUNCT = re.compile('[^\w\']+', re.UNICODE)


def _strip_punctuation(sentence, partial=False):
    """Helper function to remove punctuation"""
    if partial:
        return RE_SOME_PUNCT.sub(" ", sentence)
    else:
        return RE_ALL_PUNCT.sub(" ", sentence)


def nltk_remove_stopwords(data):
    """Helper function to remove stopwords"""
    without_stopwords = [word for word in data if word not in STOP_WORDS]
    return without_stopwords


# def is_unicode(text):
#     if isinstance(text, unicode):
#         return 1
#     elif isinstance(text, str):
#         return 0
#     else:
#         return -1
#
#
# def to_ascii(text):
#     return text.encode('ascii', 'ignore')
#
# from nltk.corpus import wordnet
#
#
# def is_english(word_to_test):
#     if wordnet.synsets(word_to_test):
#         return True
#     else:
#         return False
#
#
# if __name__ == '__main__':
#     FILE_PATH = raw_input('Enter the absolute path of '
#                           'the file you want to extract the keywords from: \n')
#     FILE_TEXT = read_file(FILE_PATH)
#     text = clean_and_format(FILE_TEXT)
#     tokens = clean_and_tokenize(text)
#     print tokens
#
#     words = []
#     for token in tokens:
#         if is_english(token.text):
#             words.append(token)
#
#     print words
#
#     words = [u'such', u'man']
#     print nltk_remove_stopwords(words)
