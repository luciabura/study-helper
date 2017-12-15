"""
Extractive summarization based on keywords genetared
by keyword extraction step in the pipeline

Al alternative approach would be domain-specific
sentence extraction.
Ideally, we want the summarized sentences to appear

"""
import math
import networkx as nx
import preprocessing.preprocessing as preprocess
from keyword_extraction.keywords_filtered import get_keywords_with_scores
from utilities.utils import read_file

WINDOW_SIZE = 2


def get_sentences_with_keywords(text, keywords):

    sentences = preprocess.sentence_tokenize(text)

    print("Sentence count: " + str(len(sentences)))

    sentences_with_keywords = {}
    for sentence in sentences:
        key_tokens = [token for token in sentence if token.text.lower() in keywords]
        if len(key_tokens) > 0:
            sentences_with_keywords[sentence] = key_tokens

    return sentences_with_keywords


def get_summary(text, sentence_num=10):
    tokens = preprocess.clean_and_tokenize(text)
    keywords_with_scores = get_keywords_with_scores(tokens)
    keywords = list(keywords_with_scores.keys())

    sentences_with_keywords = get_sentences_with_keywords(text, keywords)
    sentences = list(sentences_with_keywords.keys())
    graph = build_graph(sentences)

    add_graph_edges(graph, sentences, keywords)
    pagerank_scores = nx.pagerank(graph)

    sorted_sentences = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)
    for sentence in sorted_sentences:
        print(sentence)
        print(pagerank_scores[sentence])
    for sentence in sentences:
        if sentence in sorted_sentences[0:sentence_num]:
            print(sentence)


def get_spacy_similarity(sentence_1, sentence_2):
    return sentence_1.similarity(sentence_2)


def get_similarity(sentence_1, sentence_2):
    # Potentially better similarity score?
    common = 0
    s1_words = [token.text for token in preprocess.clean_and_tokenize(sentence_1)]
    s2_words = [token.text for token in preprocess.clean_and_tokenize(sentence_2)]
    for word in s1_words:
        if word in s2_words:
            common += 1

    if len(s1_words) and len(s2_words) > 0:
        score = common/(math.log(len(s1_words)) + math.log(len(s1_words)))
    else:
        score = 0

    return score


def add_graph_edges(graph, sentences, keywords=[]):
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


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                          'the file you want to summarize: \n')
    FILE_TEXT = read_file(FILE_PATH)
    print(get_summary(FILE_TEXT, 4))
