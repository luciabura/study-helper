import math

from keyword_extraction.keyword import Keyword, KeyPhrase


class Sentence(object):
    def __init__(self, sentence, score):
        self.score = score
        self.as_doc = sentence
        self.keywords = []
        self.key_phrases = []
        self.text = self.as_doc.text

        self.simplified_versions = []

    def compute_score(self):
        score = self.score

        for kp in self.keywords:
            div = math.log(len(self.keywords), 2)
            if div == 0:
                div = 1
            score += kp.score / div

        self.score = score

    def add_key(self, obj):
        if isinstance(obj, Keyword):
            self.keywords.append(obj)

        elif isinstance(obj, KeyPhrase):
            self.key_phrases.append(obj)

        else:
            print('Unsupported object addition!')

    def add_simplified_version(self, simplified):
        self.simplified_versions.append(simplified)