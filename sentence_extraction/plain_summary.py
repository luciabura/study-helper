"""
Extractive summarization based on keywords genetared
by keyword extraction step in the pipeline

Al alternative approach would be domain-specific
sentence extraction.
Ideally, we want the summarized sentences to appear

"""
import math
from collections import OrderedDict

import networkx as nx

import text_processing.preprocessing as preprocess
from keyword_extraction.keywords_filtered import get_keywords_with_scores

IDENTIFIER = '_A'
WINDOW_SIZE = 2


def get_sentences_with_keywords_and_scores(sentences, keywords_with_scores):
    keywords = list(keywords_with_scores.keys())

    graph = build_graph(sentences)
    add_graph_edges(graph, sentences)
    pagerank_scores = nx.pagerank(graph)

    sentences_with_keywords_and_scores = {}
    for sentence in sentences:
        key_tokens = [token for token in sentence if token.lemma_.lower() in keywords]
        if len(key_tokens) > 0:
            sentences_with_keywords_and_scores[sentence] = (pagerank_scores[sentence], key_tokens)

    return sentences_with_keywords_and_scores


def sort_scores(scores):
    sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: t[1], reverse=True))

    return sorted_scores


def get_summary(text, sentence_num=None):
    tokens = preprocess.clean_to_doc(text)
    keywords_with_scores = get_keywords_with_scores(tokens)
    sentences = preprocess.sentence_tokenize(text)
    sentences_with_keywords_and_scores = get_sentences_with_keywords_and_scores(sentences, keywords_with_scores)

    sorted_sentences = list(sort_scores(sentences_with_keywords_and_scores))

    if sentence_num is None:
        sentence_num = int(math.sqrt(len(sorted_sentences)))

    summary = []
    for sentence in sentences:
        if sentence in sorted_sentences[0:sentence_num]:
            summary.append(sentence.text.strip())

    summary = '\n'.join(summary)
    return summary


def get_spacy_similarity(sentence_1, sentence_2):
    return sentence_1.similarity(sentence_2)


def get_similarity(sentence_1, sentence_2):
    # Potentially better similarity score?
    common = 0
    s1_words = [token.text for token in preprocess.clean_to_doc(sentence_1)]
    s2_words = [token.text for token in preprocess.clean_to_doc(sentence_2)]
    for word in s1_words:
        if word in s2_words:
            common += 1

    if len(s1_words) and len(s2_words) > 0:
        score = common/(math.log(len(s1_words)) + math.log(len(s1_words)))
    else:
        score = 0

    return score


def add_graph_edges(graph, sentences):
    """
    Adds edge between all words in word sequence that are within WINDOW_SIZE
    of each other. I.e if within WINDOW_SIZE the two words co-occur
    """

    # Assume undirected graph for beginning
    for i in range(0, len(sentences) - 1):
        for j in range(i + 1, len(sentences)):
            s1 = sentences[i]
            s2 = sentences[j]
            if graph.has_node(s1) and graph.has_node(s2) and s1 != s2:
                weight_s1 = weight_s2 = get_spacy_similarity(s1, s2)
                if not graph.has_edge(s1, s2):
                    # add_edge(source, sink, weight)
                    graph.add_edge(s1, s2, weight=weight_s2)

                if not graph.has_edge(s2, s1):
                    # add_edge(source, sink, weight)
                    graph.add_edge(s2, s1, weight=weight_s1)


def get_keyphrase_score(sentence, keywords):
    count = 0
    for keyword in keywords:
        if keyword in sentence:
            count += 1

    words = sentence.split()
    if len(words) > 1:
        return count/math.log(len(words))
    else:
        return count


def build_graph(chosen_sentences):
    graph = nx.DiGraph()
    graph.add_nodes_from(chosen_sentences)

    return graph
