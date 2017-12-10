import networkx as nx
import operator

import preprocessing.preprocessing as preprocess
from utilities.utils import read_file
from utilities.words import get_cs_words

WINDOW_SIZE = 2
INCLUDE_GRAPH_POS = ['NN', 'JJ', 'NNP', 'NNS']
PUNCTUATION = ['.', '?', ',']
REFERENCE_WORDS = get_cs_words()


def get_keyword_combinations(original_tokens, scores):
    keywords = scores.keys()
    keyphrases = {}
    j = 0
    for i, _ in enumerate(original_tokens):
        if i < j:
            continue
        if original_tokens[i].lemma_.lower() in keywords:
            keyphrase_components = []
            keyphrase_length = 0
            avg_score = 0

            for token in original_tokens[i:i + 3]:
                if token.text in PUNCTUATION:
                    break

                token_lemma = token.lemma_.lower()
                token_text = token.text.lower()

                if token_lemma in keywords and token_text not in keyphrase_components:
                    keyphrase_components.append(token_text)
                    avg_score += scores[token_lemma]
                    keyphrase_length += 1
                else:
                    break

            keyphrase = ' '.join(keyphrase_components)
            keyphrases[keyphrase] = avg_score
            j = i + len(keyphrase_components)

    return keyphrases


def sort_scores(scores):
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_scores


def add_graph_edges(graph, original_tokens):
    """
    Adds edge between all words in word sequence that are within WINDOW_SIZE
    of each other. I.e if within WINDOW_SIZE the two words co-occur
    """

    # Assume undirected graph for beginning
    for i in range(0, len(original_tokens) - WINDOW_SIZE - 1):
        for j in range(i + 1, i + WINDOW_SIZE + 1):
            w1 = original_tokens[i].lemma_
            w2 = original_tokens[j].lemma_
            if graph.has_node(w1) and graph.has_node(w2) and w1 != w2:
                graph.add_edge(w1, w2, weight=1)


def build_graph(chosen_words):
    """
    Using a list of words that have been filtered to match the criteria,
    we initially build an undirected graph to run our algorithm on.
    :param chosen_words:
    :return: Undirected graph, based on the networkx library implementation
    """
    graph = nx.Graph()
    graph.add_nodes_from(chosen_words)

    return graph


def get_graph_tokens(tokens, include_filter):
    graph_tokens = [token for token in tokens
                    if token.tag_ in include_filter
                    and len(token.text) > 2]

    return graph_tokens


def get_keywords(text, keyword_count=10):
    tokens = preprocess.clean_and_tokenize(text)
    clean_tokens = preprocess.remove_stopwords(tokens)

    graph_tokens = get_graph_tokens(clean_tokens, INCLUDE_GRAPH_POS)
    keyword_count = len(graph_tokens)/3

    lemmas = [token.lemma_.lower() for token in graph_tokens]
    graph_words = list(set(lemmas))

    graph = build_graph(graph_words)
    add_graph_edges(graph, tokens)

    pagerank_scores = nx.pagerank(graph, alpha=0.85, tol=0.0001)

    keyphrases = get_keyword_combinations(tokens, pagerank_scores)

    keyphrases = [keyphrase for keyphrase, _ in sort_scores(keyphrases)]
    return keyphrases[0:keyword_count]


if __name__ == '__main__':
    FILE_PATH = raw_input('Enter the absolute path of '
                          'the file you want to extract the keywords from: \n')
    FILE_TEXT = read_file(FILE_PATH)
    print get_keywords(FILE_TEXT)
