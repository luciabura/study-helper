"""
This is the first step of the study-helper pipeline.
It return the specified (required) number of keywords from a given text
It assumes the text has not been preprocessed and it only a long string
Making use of the preprocessor file,

Using the co-occurence relation for calculating scores and weighs for words.
Other relations exist and may be inspected later for performance comparison.
window-size may be arbitrarily chosen but W=2 works best

TODO: Make sure description is accurate
"""
import networkx as nx

import text_processing.preprocessing as preprocess
from utilities.read_write import read_file

WINDOW_SIZE = 2
INCLUDE_GRAPH_POS = ['NN', 'JJ', 'NNP', 'NNS'] # 'VBG', 'VBN', 'VBD','NNPS']
INCLUDE_KEYWORD_POS = ['NN', 'NNS', 'JJ']


def get_filtered_postags(word_tag_sequence, filter_tags):
    """Returns a list of words whose tags correspond to the filter_tags
    :param word_tag_sequence : a
    :param filter_tags :
    """
    filtered_tokens = [(word, tag) for
                       word, tag in preprocess.pos_tokenize(word_tag_sequence)
                       if tag in filter_tags]

    return filtered_tokens


def get_graph_words(text_words):
    """Returns a list of words that fit the filtering criteria"""
    graph_words = [word for word, _ in get_filtered_postags(text_words, INCLUDE_GRAPH_POS)]

    return graph_words


def get_keyword_combinations(keywords, original_sequence, scores, hyphenated):
    """Returns a list of keyphrases constructed based on co-ocurrence
    :param hyphenated:
    """

    # This breaks if we won't consider original sequence with paragraphs,
    # might want to rethink structure of original_sequence
    # perfect example on which it breaks is short text with
    # a lot of newlines -> make looong keyphrases

    keyphrases = {}
    j = 0
    for i, _ in enumerate(original_sequence):
        if i < j:
            continue
        if original_sequence[i] in keywords:
            keyphrase_components = []
            keyphrase_length = 0
            avg_score = 0

            for word in original_sequence[i:i+10]:
                if word in keywords:
                    keyphrase_components.append(word)
                    avg_score += scores[word]
                    keyphrase_length += 1
                else:
                    break

            avg_score = avg_score / float(keyphrase_length)
            keyphrase = reconstruct_keyphrase(keyphrase_components, hyphenated)
            keyphrases[keyphrase] = avg_score
            keyphrases[' '.join(keyphrase_components)] = avg_score
            j = i + len(keyphrase_components)

    return keyphrases


def reconstruct_keyphrase(keyphrase_components, hyphenated):
    """Reconstructs a keyphrase from the original text based on the
    previously saved list of :param hyphenated words"""

    for i, _ in enumerate(keyphrase_components):
        if i == len(keyphrase_components) - 1:
            break

        if keyphrase_components[i]+'-'+keyphrase_components[i+1] in hyphenated:
            original_word = keyphrase_components[i]+'-'+keyphrase_components[i+1]
            keyphrase_components.pop(i)
            keyphrase_components.pop(i)
            keyphrase_components.insert(i, original_word)
    keyphrase = ' '.join(keyphrase_components)

    return keyphrase


def assign_scores_to_keywords(lemmas, scores, lemmas_to_words):
    """Returns a dictionary with keyword -> score
    This exists because we lemmatized, so we need to assign all keywords for a lemma"""

    keywords_with_scores = {}

    for lemma in lemmas:
        words = lemmas_to_words[lemma]
        for word in words:
            keywords_with_scores[word] = scores[lemma]

    return keywords_with_scores


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


def add_graph_edges(graph, word_sequence, words_to_lemmas):
    """
    Adds edge between all words in word sequence that are within WINDOW_SIZE
    of each other. I.e if within WINDOW_SIZE the two words co-occur
    """

    # Assume undirected graph for beginning
    for i in range(0, len(word_sequence) - WINDOW_SIZE - 1):
        for j in range(i + 1, i + WINDOW_SIZE + 1):
            add_graph_edge(graph, word_sequence[i], word_sequence[j], words_to_lemmas)


def add_graph_edge(graph, node_1, node_2, words_to_lemmas):
    """Adds an edge between node_1 and node_2 or weights it more if it
    already existed in the graph"""

    lem_1 = words_to_lemmas[node_1]
    lem_2 = words_to_lemmas[node_2]

    # print 'Added edge: ' + lem_1 + ' ' + lem_2
    if graph.has_node(lem_1) and graph.has_node(lem_2):
        if graph.has_edge(lem_1, lem_2):
            new_weight = graph.get_edge_data(lem_1, lem_2)['weight'] + 1
            graph.add_edge(lem_1, lem_2, weight=new_weight)
        else:
            graph.add_edge(lem_1, lem_2, weight=1)


def build_lemmas(word_sequence):
    """Returns a pair of dictionaries
    @:return word -> lemma
    @:return lemma -> word list of corresponding words"""

    lemmas_to_words = {}
    words_to_lemmas = {}
    for word in word_sequence:
        lemma = preprocess.lemmatize_word(word)
        words_to_lemmas[word] = lemma
        if lemma in lemmas_to_words:
            lemmas_to_words[lemma].append(word)
        else:
            lemmas_to_words[lemma] = [word]

    return words_to_lemmas, lemmas_to_words


def sort_keywords(scores):
    """
    :param scores: A dictionary of lemmas and their corresponding scores,
    assigned by the pagerank algorithm
    :return: The same dictionary, sorted in descending order
    """
    sorted_lemmas = [lemma for lemma in sorted(scores, key=scores.get, reverse=True)]

    return sorted_lemmas


def get_keywords(text, keyword_count=10):
    """
    :param text: String of text
    :param keyword_count: How many keywords we want to return to the caller
    :return: A list of the top keyword_count keywords in our text
    """

    # Remove all irrelevant words and hyphens, while keeping track of them for later
    text_words = preprocess.nltk_word_tokenize(text)
    (text_words, hyphenated_text_words) = preprocess.remove_hyphens(text_words)

    # Choose the words to go into the graph and begin to construct our graph
    graph_words = get_graph_words(text_words)

    words_to_lemmas, lemmas_to_words = build_lemmas(graph_words)
    lemmas = list(lemmas_to_words.keys())

    keyword_graph = build_graph(lemmas)
    add_graph_edges(keyword_graph, graph_words, words_to_lemmas)

    # Run the pagerank algorithm on the newly constructed graph
    pagerank_scores = nx.pagerank(keyword_graph, weight='weight')

    # Make sure our lemma scores get spread to their keywords
    keywords_with_scores = assign_scores_to_keywords(lemmas,
                                                     pagerank_scores,
                                                     lemmas_to_words)
    keywords = sort_keywords(keywords_with_scores)

    original_sequence = preprocess.nltk.word_tokenize(text)

    keyphrases_with_scores = get_keyword_combinations(keywords,
                                                      original_sequence,
                                                      keywords_with_scores,
                                                      hyphenated_text_words)
    keyphrases = sort_keywords(keyphrases_with_scores)

    return keyphrases[0:keyword_count]


def filter_keyphrases():
    # TODO:
    return 0


def get_word_with_rank(scores, word):
    # TODO:
    print(word + ": " + str(scores[word]));


def get_text_counts(words, word):
    print(words.count(word))


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                          'the file you want to extract the keywords from: \n')
    FILE_TEXT = read_file(FILE_PATH)
    print(get_keywords(FILE_TEXT, 30))
