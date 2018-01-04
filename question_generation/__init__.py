from spacy.attrs import *

OP = 'OP'
ANY_ALPHA = {IS_ALPHA: True, OP: "*"}
ANY = {ORTH: '', OP: '*'}


class Match:
    def __init__(self, ent_id, start_match, end_match, sentence):
        self.sentence = sentence
        self.ent_id = ent_id
        self.span = sentence[start_match:end_match]
        self.length = end_match - start_match

    def get_first_token(self):
        return self.span[0]

    def get_last_token(self):
        return self.span[-1]

    def get_sentence(self):
        return self.sentence

    def get_tokens_by_dependency(self, dependency):
        return [tok for tok in self.span if tok.dep_ == dependency]


