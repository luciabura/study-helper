import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def sentence_tokenize(text):
    return nltk.sent_tokenize(text)


def word_tokenize(text):
    """Perform the preprocessing steps on the text: remove stop words
    and punctuation, convert to lowercase, tokenize"""
    text = _strip_punctuation(text)
    data = simple_tokenize(text)
    data = [word.lower() for word in data]
    text_data = nltk.Text(data)
    words = _remove_stopwords(text_data)
    return words

def remove_hyphens(words):
    """
    :param words: A list of words, some of which may contain hyphens
    :return: The same list, but all hyphanated words are split into parts
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


def pos_tokenize(tokens):
    return nltk.pos_tag(tokens)


def simple_tokenize(text):
    return nltk.word_tokenize(text)


def lemmatize_word(word):
    return LEMMATIZER.lemmatize(word)


"""Internal functions"""

def _remove_stopwords(data):
    """Helper function to remove stopwords"""
    without_stopwords = [word for word in data if word not in STOP_WORDS]
    return without_stopwords


RE_PUNCT = re.compile('[^\w\-\']+', re.UNICODE)


def _strip_punctuation(sentence):
    """Helper function to remove punctuation"""
    return RE_PUNCT.sub(" ", sentence)
