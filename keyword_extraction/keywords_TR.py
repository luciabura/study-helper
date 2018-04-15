"""
Keyphrase extraction implementation based on the guidelines
given in the paper on TextRank by Mihalcea et al

This is
"""

import networkx as nx

import operator

import text_processing.preprocessing as preprocess
from utilities.read_write import read_file

WINDOW_SIZE = 2
INCLUDE_GRAPH_POS = ['NN', 'JJ', 'NNP', 'NNS']


def get_keyword_combinations(original_sequence, scores):
    keywords = list(scores.keys())
    keyphrases = {}
    j = 0
    for i, _ in enumerate(original_sequence):
        if i < j:
            continue
        if original_sequence[i] in keywords:
            keyphrase_components = []
            keyphrase_length = 0
            avg_score = 0

            for word in original_sequence[i:i + 3]:
                if word in keywords:
                    keyphrase_components.append(word)
                    avg_score += scores[word]
                    keyphrase_length += 1
                else:
                    break

            # avg_score = avg_score / float(keyphrase_length)
            keyphrase = ' '.join(keyphrase_components)
            keyphrases[keyphrase] = avg_score
            j = i + len(keyphrase_components)

    return keyphrases


def sort_scores(scores):
    sorted_scores = sorted(list(scores.items()), key=operator.itemgetter(1), reverse=True)

    return sorted_scores


def get_starting_scores(words):
    scores = {word: 1 for word in words}
    return scores


def add_graph_edges(graph, word_sequence):
    """
    Adds edge between all words in word sequence that are within WINDOW_SIZE
    of each other. I.e if within WINDOW_SIZE the two words co-occur
    """

    # Assume undirected graph for beginning
    for i in range(0, len(word_sequence) - WINDOW_SIZE - 1):
        for j in range(i + 1, i + WINDOW_SIZE + 1):
            w1 = word_sequence[i]
            w2 = word_sequence[j]
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


def get_graph_tokens(tokens):
    graph_tokens = [token for token in tokens
                    if token.tag_ in INCLUDE_GRAPH_POS
                    and len(token.text) > 2]

    return graph_tokens


def get_keywords(text):
    tokens = preprocess.clean_to_doc(text)
    original_sequence = [token.text.lower() for token in tokens]

    graph_tokens = get_graph_tokens(tokens)
    graph_words = [token.text.lower() for token in graph_tokens]
    # print(sorted(list(set(graph_words))))

    # Choose to display/return only a third in length
    keyword_count = int(len(graph_words)/3)

    graph = build_graph(graph_words)
    add_graph_edges(graph, original_sequence)

    # graph.remove_nodes_from(nx.isolates(graph))
    #
    # graph.graph['node']= {'shape': 'plaintext'}
    # a = drawing.nx_agraph.to_agraph(graph)
    # a.layout('dot')
    # a.draw("graph_2.png")
    # nx.draw_random(graph)
    # plt.savefig("graph.png")

    pagerank_scores = nx.pagerank(graph, alpha=0.85, tol=0.0001)

    keyphrases = get_keyword_combinations(original_sequence, pagerank_scores)
    keyphrases = [keyphrase for keyphrase, _ in sort_scores(keyphrases)]
    # print(len(graph_words))

    return keyphrases[0:keyword_count]


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                          'the file you want to extract the keywords from: \n')
    FILE_TEXT = read_file(FILE_PATH)
    print(get_keywords(FILE_TEXT))
