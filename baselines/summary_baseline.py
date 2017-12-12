import preprocessing.preprocessing as preprocess
from keyword_extraction.keywords import get_keywords
from utilities.utils import read_file


def get_summary(text):
    keywords = get_keywords(text)
    sentences = preprocess.sentence_tokenize(text)
    sentence_count = len(sentences)/3

    canditate_sentences = get_sentences_with_keywords(keywords, sentences)
    sorted_canditate_sentences = sort_sentences(canditate_sentences)

    summary = [sentence for sentence in canditate_sentences.keys() if
               sentence in sorted_canditate_sentences[0:sentence_count]]

    return summary


def sort_sentences(sentences_set):
    sorted_sentences = [sentence for sentence in
                        sorted(sentences_set, key=lambda k: len(sentences_set[k]), reverse=True)]
    return sorted_sentences


def get_sentences_with_keywords(keywords, sentences):
    sentences_with_keywords = {}

    for sentence in sentences:
        sentence_words = preprocess.nltk_word_tokenize(sentence)
        for word in sentence_words:
            if word in keywords:
                if sentences_with_keywords.has_key(sentence):
                    sentences_with_keywords[sentence].append(word)
                else:
                    sentences_with_keywords[sentence] = [word]

    return sentences_with_keywords


if __name__ == '__main__':
    FILE_PATH = raw_input('Enter the absolute path of '
                          'the file you want to summarize: \n')
    FILE_TEXT = read_file(FILE_PATH)
    summary = get_summary(FILE_TEXT)
    for sentence in summary:
        print sentence