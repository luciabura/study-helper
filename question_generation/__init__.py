from spacy.attrs import *

from utilities import NLP

OP = 'OP'
IS_ANY_TOKEN = NLP.vocab.add_flag(lambda x: True)
ANY_TOKEN = {OP: '*', IS_ANY_TOKEN: True}
NO_TOKEN = {OP: '!', IS_ANY_TOKEN: True}
ANY_ALPHA = {IS_ALPHA: True, OP: "*"}

# IS_SUBJECT = NLP.add_flag(lambda x: "subj" in x.dep_)
# SUBJECT = {IS_SUBJECT: True}


class Match(object):
    def __init__(self, ent_id, start_match, end_match, sentence):
        self.sentence = sentence
        self.ent_id = ent_id
        self.span = sentence[start_match:end_match]
        self.length = end_match - start_match

    def get_first_token(self):
        return self.span[0]

    def get_last_token(self):
        return self.span[-1]

    def get_tokens_by_attributes(self, dependency=None, tag=None, pos=None, head=None):
        tokens = []

        for tok in self.span:
            if(
                (dependency is None or tok.dep_ == dependency)
                and (head is None or tok.head == head)
                and (pos is None or tok.pos_ == pos)
                and (tag is None or tok.tag_ == tag)
            ):
                tokens.append(tok)

        if not tokens:
            return None

        return tokens

    def get_token_by_attributes(self, dependency=None, tag=None, pos=None, head=None):
        tokens = self.get_tokens_by_attributes(dependency, tag, pos, head)

        if not tokens:
            return None

        return tokens[0]
