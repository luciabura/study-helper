from collections import OrderedDict

import sys

import math
import spacy
from spacy import displacy

from keyword_extraction.keywords_filtered import get_keywords_with_scores
from preprocessing import preprocessing as preprocess
from summarization.summary import get_sentences_with_keywords_and_scores
from data import TEST_TEXT

NLP = spacy.load('en_core_web_md')
INFINITY = math.inf


def sort_scores(scores):
    sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: t[1], reverse=True))

    return sorted_scores


def generate_questions(text):

    # The protocol for getting sentences with corresponding scores and keywords
    tokens = preprocess.clean_and_tokenize(text)
    keywords_with_scores = get_keywords_with_scores(tokens)
    sentences = preprocess.sentence_tokenize(text)
    sentences_with_keywords_and_scores = get_sentences_with_keywords_and_scores(sentences, keywords_with_scores)

    sorted_sentences = list(sort_scores(sentences_with_keywords_and_scores))
    for sentence in sorted_sentences:
        for token in sentence:
            print(token.text, token.dep_)
            # showTree(sentence)
        print(sentences_with_keywords_and_scores[sentence][1])
        break


def has_pronouns(sentence):
    for word in sentence:
        if word.tag_ == 'PRP' or word.tag_ == 'PRP$':
            return True

    return False


def get_coreference(pronoun):
    """Implement if time -> pronoun get document coreference with neuralcoref"""
    return 0


def show_dependencies(sentence):
    displacy.serve(sentence, style='dep', port=5001)


def get_phrase(token):
    phrase = []
    min_index = INFINITY
    max_index = -1

    for child in token.subtree:
        if child.dep_.endswith("mod") or child.dep_ == "compound" or child == token or child.dep_ == "poss" or child.dep_ == "case":
            if min_index > child.i:
                min_index = child.i

            if max_index < child.i:
                max_index = child.i

            phrase.append(child)

    return phrase, min_index, max_index


# def get_subject()

def choose_wh_word(token):
    """
        Takes a token and returns the corresponding wh-word to construct a sentence with it
    """
    return 0


def mark_unmovable_phrase(sentence):
    """

    :param sentence:
    :return:
    """
    return 0


def is_valid_sentence(sentence):
    """

    :param sentence:
    :return:
    """
    return 0


def first_question():
    text = NLP(u'Computer architecture is a set of rules that describe the functionality of computer systems.')
    text_2 = NLP(u"Apple's logo was designed for Steve Jobs.")
    text_3 = NLP(u"Stacy went to the store to see Johnny.")

    show_dependencies(text_3)
    # for nc in text.noun_chunks:
    #     print(nc)

    # print(get_phrase(text[7]))


if __name__ == '__main__':
    # generate_questions(TEST_TEXT)
    first_question()