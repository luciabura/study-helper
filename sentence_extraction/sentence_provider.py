"""
Extractive summarization based on keywords genetared
by keyword extraction step in the pipeline

Al alternative approach would be domain-specific
sentence extraction.
Ideally, we want the summarized sentences to appear

"""
import math
import networkx as nx

from keyword_extraction.keyword_provider import KeywordProvider
from sentence_extraction.sentence import Sentence
from text_processing import preprocessing as preprocess
from utilities.read_write import read_file

IDENTIFIER = '_C'
WINDOW_SIZE = 2


class SentenceProvider(object):
    def __init__(self, doc, topic=None):
        self.keyword_provider = KeywordProvider(doc, topic)
        self.keywords = self.keyword_provider.keywords
        self.key_phrases = self.keyword_provider.key_phrases

        self.sentences = self.keyword_provider.sentences
        self.sentence_objects = []

        # Expect token topic is used witse
        self.topic = topic

        self._compute_sentences()

    def _compute_sentences(self):
        graph = self._build_graph(self.sentences)
        sentence_graph = self._add_graph_edges(graph, self.sentences, self._spacy_similarity)
        pagerank_scores = self._get_pagerank_scores(sentence_graph)

        sentences_dict = self._get_sentences_with_scores(pagerank_scores, self.sentences)

        sentences_dict = self._augment_sentence_objects(sentences_dict, self.keyword_provider.keywords)
        sentences_dict = self._augment_sentence_objects(sentences_dict, self.keyword_provider.key_phrases)

        self._calculate_final_scores(sentences_dict)

        self.sentence_objects = self._sort_by_score(list(sentences_dict.values()), descending=True)

    def get_top_sentences(self, sentence_count=None, trim=True):
        if sentence_count is None:
            sentence_count = math.ceil(len(self.sentences)/11)

        # if sentence_count < 2:
            # print('Please provide more text for an accurate summary.')

        if trim:
            return self.sentence_objects[0:sentence_count]
        else:
            return self.sentence_objects

    @staticmethod
    def _calculate_final_scores(sentence_dictionary):
        for sent in sentence_dictionary.values():
            sent.compute_score()

    @staticmethod
    def _sort_by_score(unsorted, descending=False):
        return sorted(unsorted, key=lambda el: el.score, reverse=descending)

    @staticmethod
    def _augment_sentence_objects(sentences_with_scores, objects):
        for obj in objects:
            sentence = obj.sentence
            if sentence in sentences_with_scores:
                sentences_with_scores[sentence].add_key(obj)

        return sentences_with_scores

    @staticmethod
    def _get_sentences_with_scores(pagerank_scores, sentences):
        sentences_with_scores = {}
        for sentence in sentences:
            sent_object = Sentence(sentence, pagerank_scores[sentence])
            sentences_with_scores[sentence] = sent_object

        return sentences_with_scores

    @staticmethod
    def _get_pagerank_scores(graph):
        pagerank_scores = nx.pagerank(graph)
        return pagerank_scores

    @staticmethod
    def _build_graph(chosen_sentences):
        graph = nx.DiGraph()
        graph.add_nodes_from(chosen_sentences)

        return graph

    @staticmethod
    def _spacy_similarity(sentence_1, sentence_2):
        # print("S1:{}\nS2:{}\nSim:{}\n".format(sentence_1, sentence_2, sentence_1.similarity(sentence_2)))
        return sentence_1.similarity(sentence_2)

    @staticmethod
    def _add_graph_edges(graph, sentences, similarity):
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
                    weight_s1 = weight_s2 = similarity(s1, s2)
                    if not graph.has_edge(s1, s2):
                        # add_edge(source, sink, weight)
                        graph.add_edge(s1, s2, weight=weight_s2)

                    if not graph.has_edge(s2, s1):
                        # add_edge(source, sink, weight)
                        graph.add_edge(s2, s1, weight=weight_s1)

        return graph

    def get_summary(self, sentence_count=None, trim=True):

        top_sentences_text = [sent.text for sent in self.get_top_sentences(sentence_count, trim)]
        sentences_text = [sent.text for sent in self.sentences]

        summary = []

        for sent in sentences_text:
            if sent in top_sentences_text:
                summary.append(sent)

        summary = '\n'.join(summary)
        return summary


def get_summary(file_text, sentence_count=None, trim=True):
    d = preprocess.clean_to_doc(file_text)
    sentence_prov = SentenceProvider(d)

    return sentence_prov.get_summary(sentence_count, trim)


if __name__ == '__main__':
    FILE_PATH = input('Enter the absolute path of '
                      'the file you want to summarize: \n')
    # OUTPUT_DIR = input('Directory to put summary in: \n')
    FILE_TEXT = read_file(FILE_PATH)

    document = preprocess.clean_to_doc(FILE_TEXT)
    summarizer = SentenceProvider(document)

    # topic = NLP(input("Topic?: "))
    # summarizer_topic = SentenceProvider(document, topic=topic)

    print(summarizer.get_summary())

    # print("")
    # print(summarizer_topic.get_summary())
    # print(summarizer.get_summary())

    # print_summary_to_file(get_summary, FILE_PATH, OUTPUT_DIR, IDENTIFIER)

    # for keyword in summarizer.keyword_provider.keywords:
    #     print(keyword.text, keyword.score)

    # for sentence in summarizer.top_sentences:
    #     print(sentence.text)
        # print(sentence.score)
        # for keyword in sentence.keywords:
        #     print(keyword.text)
