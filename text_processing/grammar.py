from spacy import displacy

from utilities import INFINITY, NLP


def has_pronouns(span):
    for word in span:
        if word.tag_ == 'PRP' or word.tag_ == 'PRP$':
            return True

    return False


def extract_noun_phrase(token, sentence, exclude_span=None, include_span=None, discard_punct=None):
    # start_index = INFINITY
    # end_index = -1

    # for child in token.subtree:
    #     """This will fail in some cases, might want to try to just get full subtree, but then need to pay attention
    #     what we call it on. For now, I'm going to call it only on subjects and object so should be OK to get subtree"""
    #     # if exclude_span and child in exclude_span:
    #     #     continue
    #     if include_span and child not in include_span:
    #         continue
    #     elif discard_punct and child.text in discard_punct:
    #         continue
    #
    #     if start_index > child.i:
    #         start_index = child.i
    #
    #     if end_index < child.i:
    #         end_index = child.i

    subtree_span = get_subtree_span(token, sentence)

    np_tokens = [tok for tok in subtree_span]

    if exclude_span:
        np_tokens = list(filter(lambda x: x not in exclude_span, np_tokens))

    if discard_punct:
        np_tokens = list(filter(lambda x: x.text not in discard_punct, np_tokens))

    return np_tokens


def get_verb_phrase(token, sentence):
    start_index = INFINITY
    end_index = -1
    # TODO: Cover case with 'come up with...' etc up is 
    for child in token.subtree:
        if (child.dep_.startswith("aux") and child.head == token) \
                or (child.dep_ == 'neg' and child.head == token) \
                or (child.dep_ == 'prt' and child.head == token) \
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
    # This assumes sentence is passes as doc
    sent_span = sentence[0:]

    if len(sent_span) == 0:
        return False

    if sent_span.root.pos_ != "VERB":
        return False

    if not any(token.dep_.startswith("nsubj") and token.head.dep_ == "ROOT" for token in sentence):
        return False

    return True


def is_vowel(character):
    return character.lower() in ["a", "e", "i", "o", "u"]


def is_past_tense(token):
    return token.tag_ == 'VBD' \
           # or token.tag_ == 'VBN'


def is_3rd_person(verb):
    return verb.tag_ == 'VBZ'


def is_valid_subject(subject):
    """If it has any coreference or if it contains something like those, this etc, then it isn't"""
    if any(tok.pos_ == 'PRON' for tok in subject):
        return False

    return True


def safe_join(sentence_components):
    to_replace = [" ,", " .", " - ", " '", " ;", " :", " !", " ?"]
    safe = ' '.join(sentence_components)

    for tok in to_replace:
        if tok in safe:
            new_tok = tok.replace(' ', '')
            safe = safe.replace(tok, new_tok)

    return safe


def show_dependencies(sentence, port=5000):
    displacy.serve(sentence, style='dep', port=port)


def is_plural(token):
    return token.tag_ == 'NNS' or token.tag_ == 'NNPS'


def print_noun_chunks(sentence):
    for chunk in sentence.noun_chunks:
        print(chunk)


def find_parent_verb(token, depth=1):
    if token.dep_ == 'ROOT' and token.pos_ != 'VERB':
        return None

    elif token.dep_ == 'ROOT' or (token.pos_ == 'VERB' and depth == 0):
        return token

    else:
        if depth > 0:
            depth -= 1
        return find_parent_verb(token.head, depth)


def get_verb_correct_tense(dependant_noun_phrase, dependant_verb, verb_lemma=''):
    past_tense = is_past_tense(dependant_verb)
    if verb_lemma == 'be':
        if any(is_plural(tok) for tok in dependant_noun_phrase):
            if past_tense:
                verb = 'were'
            else:
                verb = u'are'
        else:
            if past_tense:
                verb = 'was'
            else:
                verb = u'is'
        return verb
    else:
        return None


def remove_spans(sentence, spans):
    sentence_text = []
    for tok in sentence:
        if any(tok in span for span in spans):
            continue
        sentence_text.append(tok.text)
    # sentence = NLP(' '.join(sentence_text))
    return sentence_text


def get_subtree_span(token, sentence):
    start_index = INFINITY
    end_index = -1
    for child in token.subtree:
        if start_index > child.i:
            start_index = child.i

        if end_index < child.i:
            end_index = child.i

    return sentence[start_index: (end_index + 1)]


def spacy_similarity(text1, text2):
    doc1 = NLP(text1)
    doc2 = NLP(text2)
    return doc1.similarity(doc2)