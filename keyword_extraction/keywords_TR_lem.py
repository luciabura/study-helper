import networkx as nx
from spacy.lang.en.stop_words import STOP_WORDS

import text_processing.preprocessing as preprocess
from utilities import NLP
from utilities.read_write import read_file

# from utilities.words import get_cs_words

WINDOW_SIZE = 2
INCLUDE_GRAPH_POS = ['NN', 'JJ', 'NNP', 'NNS', 'NNPS']


# REFERENCE_WORDS = get_cs_words()


class Keyword(object):

    def __init__(self, token, score, sentence):
        self.token = token
        self.text = self.token.lower_
        self.score = score
        self.sentence = sentence
        self.doc = token.doc


class KeyPhrase(object):

    def __init__(self, start_index, end_index, sentence, keywords):
        self.span = sentence[start_index:end_index + 1]
        self.length = len(self.span)
        self.text = self.span.string.strip().lower()
        self.keywords = keywords
        self.sentence = sentence
        self.score = self.__calculate_score()

    def __calculate_score(self):
        score = 0
        for key in self.keywords:
            score += key.score
        return score

    def similarity(self, token):
        return token.similarity(self.span)


class KeywordProvider(object):

    def __init__(self, doc, topic=None):
        self.tokens = [token for token in doc]
        self.sentences = [sent.as_doc() for sent in doc.sents]
        self.doc = doc
        self.keywords = []
        self.key_phrases = []

        # Expect token topic
        self.topic = topic

        self.__compute_keywords()

    def __compute_keywords(self):
        # Build the keyword graph
        graph_tokens = get_graph_tokens(self.tokens, INCLUDE_GRAPH_POS)

        graph_words = get_graph_words(graph_tokens)
        graph = build_graph(graph_words)
        keyword_graph = add_graph_edges(graph, self.tokens)

        if self.topic:
            add_graph_weights(keyword_graph, topic=self.topic)

        # Run pagerank
        pagerank_scores = get_pagerank_scores(keyword_graph)

        keywords_with_scores = get_keywords_with_scores(pagerank_scores, self.sentences)
        self.keywords = sort_by_score(keywords_with_scores, descending=True)

        # if self.topic:
        #     self.re_giggle_scores(self.keywords, self.topic)

        key_phrases_with_scores = get_keyword_combinations(keywords_with_scores, self.sentences)
        self.key_phrases = sort_by_score(key_phrases_with_scores, descending=True)

    def show_keywords(self, keyword_count=None, trim=True):
        if keyword_count is None:
            keyword_count = int(len(self.key_phrases) / 3)

        keyword_list = []
        for keyword in self.keywords:
            if keyword.text not in keyword_list:
                keyword_list.append(keyword.text)

        if trim:
            return keyword_list[0:keyword_count]
        else:
            return keyword_list

    def show_key_phrases(self, key_phrase_count=None, trim=True, filter_similar=True):
        if key_phrase_count is None:
            key_phrase_count = int(len(self.key_phrases) / 3)

        key_phrase_list = []

        if filter_similar:
            _key_phrases = filter_similar_key_phrases(self.key_phrases)
        else:
            _key_phrases = self.key_phrases

        for _key_phrase in _key_phrases:
            if _key_phrase.text not in key_phrase_list:
                key_phrase_list.append(_key_phrase.text)

        if trim:
            return key_phrase_list[0:key_phrase_count]
        else:
            return key_phrase_list

    @staticmethod
    def re_giggle_scores(keywords, topic):
        for keyword in keywords:
            keyword.score += keyword.token.similarity(topic)


def sort_by_score(unsorted, descending=False):
    return sorted(unsorted, key=lambda el: el.score, reverse=descending)


def get_keyword_combinations(keyword_list, sentences):
    keyword_tokens = {}

    for kw in keyword_list:
        keyword_tokens[kw.token] = kw

    key_phrases = []

    for sentence in sentences:
        j = 0
        for i, token in enumerate(sentence):
            if i < j:
                continue
            if token in keyword_tokens:
                start_index = token.i
                end_index = token.i
                keyword_objects = [keyword_tokens[token]]

                for tok in sentence[i:i + 4]:
                    if tok.pos_ == 'PUNCT':
                        if tok.tag_ == 'HYPH':
                            end_index = tok.i + 1
                            continue
                        else:
                            break

                    if tok in keyword_tokens:
                        end_index = tok.i
                        keyword_objects.append(keyword_tokens[tok])
                    else:
                        break

                key_phrase = KeyPhrase(start_index, end_index, sentence, keyword_objects)
                key_phrases.append(key_phrase)

                j = i + (end_index - start_index + 1)

    return key_phrases


def get_graph_words(tokens):
    lemmas = [token.lemma_.lower() for token in tokens]
    graph_words = list(set(lemmas))

    return graph_words


def add_graph_edges(graph, tokens, directed=False):
    """
    Adds edge between all words in word sequence that are within WINDOW_SIZE
    of each other. I.e if within WINDOW_SIZE the two words co-occur
    Build an undirected graph
    """
    # Assume undirected graph for beginning
    for i in range(0, len(tokens) - WINDOW_SIZE - 1):
        for j in range(i + 1, i + WINDOW_SIZE + 1):
            w1 = tokens[i].lemma_
            w2 = tokens[j].lemma_
            if graph.has_node(w1) and graph.has_node(w2) and w1 != w2:
                graph.add_edge(w1, w2, weight=1)

                if directed:
                    graph.add_edge(w2, w1, weight=1)

    return graph


def add_graph_weights(graph, topic=None, domain_words=None):
    for node in graph.nodes:
        node_token = NLP(node)
        # num_neighbours = len(graph.neighbors(node))
        for neighbour in graph.neighbors(node):
            graph[neighbour][node]['weight'] += node_token.similarity(topic)


def print_graph(graph):
    # graph.remove_nodes_from(nx.isolates(graph))
    graph.graph['node'] = {'shape': 'plaintext'}
    a = nx.drawing.nx_agraph.to_agraph(graph)
    a.layout('dot')
    a.draw("graph_TR_LEM_2.png")


def build_graph(graph_words, directed=False):
    """
    Using a list of words that have been filtered to match the criteria,
    we initially build an undirected graph to run our algorithm on.
    :param graph_words:
    :return: Undirected graph, based on the networkx library implementation
    """
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    graph.add_nodes_from(graph_words)

    return graph


def get_graph_tokens(tokens, include_filter):
    graph_tokens = [token for token in tokens
                    if token.tag_ in include_filter
                    and token.text not in STOP_WORDS]

    # print([(gt.tag_, gt.text) for gt in graph_tokens])

    return graph_tokens


def get_pagerank_scores(graph):
    pagerank_scores = nx.pagerank(graph, alpha=0.85, tol=0.0001)

    return pagerank_scores


def get_keywords_with_scores(pagerank_scores, sentences):
    keywords_with_scores = []
    for sentence in sentences:
        for token in sentence:
            if token.lemma_ in pagerank_scores.keys():
                keyword_with_score = Keyword(token, score=pagerank_scores[token.lemma_], sentence=sentence)
                keywords_with_scores.append(keyword_with_score)

    keywords_with_scores.sort(key=lambda kw: kw.score, reverse=True)

    return keywords_with_scores


def filter_similar_key_phrases(sorted_keyphrases):
    key_phrase_list = []
    for kp in sorted_keyphrases:
        seen = False
        for existing_kp in key_phrase_list:
            if set(kp.text.split()) < set(existing_kp.text.split()):
                seen = True
                break
            elif set(existing_kp.text.split()) < set(kp.text.split()):
                key_phrase_list.remove(existing_kp)
                break
            else:
                # Checking for subsets of token lemmas
                # -> eg: minimal generating sets, minimal set will not include 'minimat set' in final keyphrases

                set_k_tokens = set([tok.lemma_ for tok in kp.span])
                set_e_tokens = set([tok.lemma_ for tok in existing_kp.span])

                # exclude words separated by hyphens 
                if '-' in set_k_tokens and '-' in set_e_tokens \
                        or '-' not in set_k_tokens and '-' not in set_e_tokens:
                    if set_k_tokens <= set_e_tokens:
                        seen = True
                        break
                    elif set_e_tokens <= set_k_tokens:
                        key_phrase_list.remove(existing_kp)

        if not seen:
            key_phrase_list.append(kp)

    return key_phrase_list


class OriginalKeywordProvider(KeywordProvider):
    def __init__(self, doc):
        KeywordProvider.__init__(self, doc)

    @staticmethod
    def get_graph_words(tokens):
        graph_words = list(set([token.lower_ for token in tokens]))

        return graph_words

    @staticmethod
    def add_graph_edges(graph, tokens):
        for i in range(0, len(tokens) - WINDOW_SIZE - 1):
            for j in range(i + 1, i + WINDOW_SIZE + 1):
                w1 = tokens[i].lower_
                w2 = tokens[j].lower_
                if graph.has_node(w1) and graph.has_node(w2) and w1 != w2:
                    graph.add_edge(w1, w2, weight=1)

        return graph

    @staticmethod
    def get_keywords_with_scores(pagerank_scores, sentences):
        keywords_with_scores = []
        for sentence in sentences:
            for token in sentence:
                if token.lower_ in pagerank_scores.keys():
                    keyword_with_score = Keyword(token, score=pagerank_scores[token.lower_], sentence=sentence)
                    keywords_with_scores.append(keyword_with_score)

        keywords_with_scores.sort(key=lambda kw: kw.score, reverse=True)

        return keywords_with_scores


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                      'the file you want to extract the keywords from: \n')
    FILE_TEXT = read_file(FILE_PATH)
    document = preprocess.clean_to_doc(FILE_TEXT)
    lemma_provider = KeywordProvider(document)

    science = preprocess.clean_to_doc("systems")
    lemma_provider_topic = KeywordProvider(document, science)

    print("\nWithout topic")
    key_phrases = lemma_provider.key_phrases
    for key_phrase in key_phrases:
        print("Key-phrase: {}, Score: {}".format(key_phrase.text, key_phrase.score))

    print("\nWith topic")
    key_phrases = lemma_provider_topic.key_phrases
    for key_phrase in key_phrases:
        print("Key-phrase: {}, Score: {}".format(key_phrase.text, key_phrase.score))
    # original_provider = OriginalKeywordProvider(document)

    # print('\n Keyphrases:')
    #
    # key_phrases = lemma_provider.show_key_phrases(trim=True, filter_similar=True)
    # print(key_phrases)
    #
    # key_phrases = original_provider.show_key_phrases(trim=False, filter_similar=False)
    # print(key_phrases)

    # topic = NLP("human-computer interaction")
    #
    # kps = lemma_provider.key_phrases
    # for kp in kps:
    #     print(kp.similarity(topic))
