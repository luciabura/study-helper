import networkx as nx
from spacy.lang.en.stop_words import STOP_WORDS

import text_processing.preprocessing as preprocess
from utilities.read_write import read_file

# from utilities.words import get_cs_words

WINDOW_SIZE = 2
INCLUDE_GRAPH_POS = ['NN', 'JJ', 'NNP', 'NNS', 'NNPS']


# REFERENCE_WORDS = get_cs_words()


def get_keyword_combinations(original_tokens, scores):
    global preposition
    keywords = list(scores.keys())
    keyphrases_with_scores = {}

    j = 0
    for i, _ in enumerate(original_tokens):
        if i < j:
            continue
        if original_tokens[i].lemma_.lower() in keywords:
            keyphrase_components = []
            keyphrase_tokens = []

            keyphrase_length = 0
            avg_score = 0

            skip_count = 0

            # preposition = None
            # preposition_pos = None

            for token in original_tokens[i:i + 4]:
                if skip_count > 0:
                    skip_count -= 1
                    continue

                if token.pos_ == 'PUNCT':
                    # break
                    if token.tag_ == 'HYPH':
                        second_part = original_tokens[token.i + 1]
                        if second_part.lower_ not in keyphrase_components and second_part.lemma_.lower() in keywords:
                            first_part = keyphrase_components.pop(keyphrase_length - 1)
                            keyphrase_components.append(''.join([first_part, '-', second_part.lower_]))
                            keyphrase_tokens.append(second_part)
                            avg_score += scores[token.head.lemma_.lower()]
                            skip_count = 1
                            continue
                        else:
                            break
                    else:
                        break

                token_lemma = token.lemma_.lower()
                token_text = token.text.lower()

                if token_lemma in keywords and token_text not in keyphrase_components:

                    # if preposition and token.head == preposition:
                    #     keyphrase_components.insert(preposition_pos, preposition.lower_)
                    #     keyphrase_length += 1

                    keyphrase_components.append(token_text)
                    keyphrase_tokens.append(token)
                    avg_score += scores[token_lemma]
                    keyphrase_length += 1
                else:
                    if token_text in keyphrase_components:
                        continue
                    else:
                        break

                    # if token.dep_ == 'prep':
                    #     print(token, keyphrase_length)
                    #     preposition = token
                    #     preposition_pos = keyphrase_length
                    # else:
                    #     break

            keyphrase = ' '.join(keyphrase_components)
            keyphrases_with_scores[keyphrase] = (avg_score, keyphrase_tokens)
            j = i + len(keyphrase_components)

    return keyphrases_with_scores


def sort_scores(scores):
    sorted_scores = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)

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
                    and token.text not in STOP_WORDS]

    return graph_tokens


def get_keyphrases_with_scores(text):
    tokens = preprocess.clean_and_tokenize(text)
    clean_tokens = preprocess.remove_stopwords(tokens)

    graph_tokens = get_graph_tokens(clean_tokens, INCLUDE_GRAPH_POS)

    lemmas = [token.lemma_.lower() for token in graph_tokens]
    graph_words = list(set(lemmas))

    graph = build_graph(graph_words)
    add_graph_edges(graph, tokens)

    # graph.remove_nodes_from(nx.isolates(graph))

    # graph.graph['node'] = {'shape': 'plaintext'}
    # a = nx.drawing.nx_agraph.to_agraph(graph)
    # a.layout('dot')
    # a.draw("graph_TR_LEM_2.png")

    pagerank_scores = nx.pagerank(graph, alpha=0.85, tol=0.0001)

    keyphrases_with_scores = get_keyword_combinations(tokens, pagerank_scores)

    return keyphrases_with_scores


def get_keywords(text, keyword_count=10, customize_count=False, trim=True, filter_similar=False):
    keyphrases_with_scores = get_keyphrases_with_scores(text)

    if customize_count is False:
        keyword_count = int(len(keyphrases_with_scores) / 3)

    sorted_keyphrases = [keyphrase for keyphrase, _ in sort_scores(keyphrases_with_scores)]

    if filter_similar:
        return get_filtered_keywords(sorted_keyphrases, keyphrases_with_scores)

    if trim:
        return sorted_keyphrases[0:keyword_count]
    else:
        return sorted_keyphrases


def get_filtered_keywords(sorted_keyphrases, keyphrases_with_scores):
    """Assumes keyphrases comes in an order of length, i.e that score is cummulative
    If this assumption no longer holds, this method is wrong/incomplete"""
    keyphrases = []
    for keyphrase in sorted_keyphrases:
        seen = False
        for existing_keyphrase in keyphrases:
            if set(keyphrase.split()) < set(existing_keyphrase.split()):
                seen = True
            elif set(existing_keyphrase.split()) < set(keyphrase.split()):
                keyphrases.remove(existing_keyphrase)
                break
            else:
                # Checking for subsets of token lemmas
                # -> eg: minimal generating sets, minimal set will not include 'minimat set' in final keyphrases

                set_k_tokens = set([tok.lemma_.lower() for tok in keyphrases_with_scores[keyphrase][1]])
                set_e_tokens = set([tok.lemma_.lower() for tok in keyphrases_with_scores[existing_keyphrase][1]])

                if ('-' in keyphrase and '-' in existing_keyphrase) \
                        or ('-' not in keyphrase and '-' not in existing_keyphrase):
                    if set_k_tokens < set_e_tokens:
                        seen = True
                    elif set_e_tokens < set_k_tokens:
                        keyphrases.remove(existing_keyphrase)

        if not seen:
            keyphrases.append(keyphrase)

    return keyphrases


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                      'the file you want to extract the keywords from: \n')
    FILE_TEXT = read_file(FILE_PATH)
    print((get_keywords(FILE_TEXT, trim=True, filter_similar=True)))
