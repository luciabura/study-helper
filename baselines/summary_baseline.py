import math

import text_processing.preprocessing as preprocess
from keyword_extraction.keywords_v1 import get_keywords
from utilities.read_write import read_file


def get_summary(text, sentence_count=None):
    keywords = get_keywords(text)
    sentences = preprocess.nltk_sentence_tokenize(text)

    if sentence_count is None:
        sentence_count = math.ceil(len(sentences)/5)

    canditate_sentences = get_sentences_with_keywords(keywords, sentences)
    sorted_canditate_sentences = sort_sentences(canditate_sentences)

    summary = [sentence for sentence in sentences if
               sentence in sorted_canditate_sentences[0:sentence_count]]

    summary_text = '\n'.join(summary)

    return summary_text


def sort_sentences(sentences_set):
    sorted_sentences = [sentence for sentence in
                        sorted(sentences_set,
                               key=lambda k: sentences_set[k],
                               reverse=True)]
    return sorted_sentences


def get_sentences_with_keywords(keywords, sentences):
    sentences_with_keywords = {}

    for sentence in sentences:
        sentence_words = preprocess.nltk_word_tokenize(sentence)
        for word in sentence_words:
            if word in keywords:
                if sentence in sentences_with_keywords:
                    sentences_with_keywords[sentence] += 1
                else:
                    sentences_with_keywords[sentence] = 1
                sentences_with_keywords[sentence] = sentences_with_keywords[sentence] * 1.0 \
                                                    / math.log(len(sentence_words), 2)

    return sentences_with_keywords


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                          'the file you want to summarize: \n')
    FILE_TEXT = read_file(FILE_PATH)
    summary = get_summary(FILE_TEXT)
    for sentence in summary:
        print(sentence)
