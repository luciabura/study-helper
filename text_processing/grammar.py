from spacy import displacy

from utilities import INFINITY


def has_pronouns(span):
    for word in span:
        if word.tag_ == 'PRP' or word.tag_ == 'PRP$':
            return True

    return False


def extract_noun_phrase(token, sentence, exclude_span=None):
    start_index = INFINITY
    end_index = -1

    for child in token.subtree:
        """This will fail in some cases, might want to try to just get full subtree, but then need to pay attention 
        what we call it on. For now, I'm going to call it only on subjects and object so should be OK to get subtree"""
        if exclude_span and child in exclude_span:
            continue
        # if child.dep_.endswith("mod") \
        #         or child.dep_ == "compound" \
        #         or child == token \
        #         or child.dep_ == "poss" \
        #         or child.dep_ == "case":

        if start_index > child.i:
            start_index = child.i

        if end_index < child.i:
            end_index = child.i

    return sentence[start_index: (end_index + 1)]


def get_verb_phrase(token, sentence):
    start_index = INFINITY
    end_index = -1
    # TODO: Cover case with 'come up with...' etc up is 
    for child in token.subtree:
        if (child.dep_.startswith("aux") and child.head == token) \
                or (child.dep_ == 'neg' and child.head == token) \
                or child == token:
            if start_index > child.i:
                start_index = child.i

            if end_index < child.i:
                end_index = child.i

    return sentence[start_index: (end_index + 1)]


def is_valid_sentence(sentence):
    """

    :param sentence:
    :return:
    """
    return True


def is_past_tense(token):
    return token.tag_ == 'VBD' or token.tag_ == 'VBN'


def is_3rd_person(verb):
    return verb.tag_ == 'VBZ'


def is_valid_subject(subject):
    """If it has any coreference or if it contains something like those, this etc, then it isn't"""
    if any(tok.pos_ == 'PRON' for tok in subject):
        return False

    return True


def show_dependencies(sentence, port=5000):
    displacy.serve(sentence, style='dep', port=port)