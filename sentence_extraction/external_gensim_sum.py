from gensim.summarization import summarize

from text_processing import preprocessing as preprocess

IDENTIFIER = '_B'


def get_summary(text):
    text = preprocess.clean_and_format(text)
    return summarize(text, ratio=0.1)
