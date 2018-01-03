import spacy
import sys
import re
from spacy import displacy
from spacy.attrs import DEP, POS, TAG, ORTH
from spacy.matcher import Matcher

from question_generation.sentence_simplifier import PARSER

NLP = spacy.load('en_core_web_md')
APPOS = Matcher(NLP.vocab)


def is_plural(token):
    return token.tag_ == 'NNS' or token.tag_ == 'NNPS'


def show_tree(sent):
    def __show_tree(token, level):
        tab = "\t" * level
        sys.stdout.write("\n%s{" % (tab))
        [__show_tree(t, level + 1) for t in token.lefts]
        sys.stdout.write("\n%s\t%s [%s] (%s)" % (tab, token, token.dep_, token.tag_))
        [__show_tree(t, level + 1) for t in token.rights]
        sys.stdout.write("\n%s}" % (tab))

    return __show_tree(sent.root, 1)


def find_root(sentence):
    for tok in sentence:
        if tok.dep_ == 'ROOT':
            return tok


def get_stanford_tree(sentence):
    tagged_sentence = [(token.text, token.tag_) for token in sentence]
    t = PARSER.tagged_parse(tagged_sentence)
    return t


def print_noun_chunks(sentence):
    for chunk in sentence.noun_chunks:
        print(chunk)


def extract_subordinate(sentence):
    for tok in sentence:
        if tok.dep_ == 'advcl':
            for sub in tok.subtree:
                print(sub)


def appositive_sentence(appositive, dependant):
    if is_plural(dependant):
        verb = u'are'
    else:
        verb = u'is'

    # Need depentand noun phrase to construct sentence
    components = [tok.text for tok in dependant]
    components.append(verb)
    components.extend([tok.text for tok in appositive])

    s = ' '.join(components)
    s += '.'

    sentence = NLP(s)
    return sentence


def extract_appositives(sentence):
    matches = APPOS(sentence)

    for ent_id, start, end in matches:
        span = sentence[start:end]
        # First token is our noun_phrase_0
        np_0 = span[0]
        print(np_0)
        appositive = []
        for w in np_0.subtree:
            appositive.append(w)

    print(appositive)
    dependant = np_0.head
    dep_noun_phrase = [np for np in sentence.noun_chunks if dependant in np]
    sent_appos = appositive_sentence(appositive, dep_noun_phrase)
    print(sent_appos)
    return sentence, appositive


def initialize_matchers():
    pattern = [{DEP: 'appos'}]
    APPOS.add("Appositive", None, pattern)


if __name__ == '__main__':
    doc = NLP(
        'Selling snowballed because of waves of automatic stop-loss orders, which are triggered by computer when prices fall to certain levels')
    doc2 = NLP('John studied, hoping to get a good grade.')
    doc5 = NLP('As John slept, I studied.')
    doc3 = NLP('A computer is a high speed machine.')
    doc4 = NLP('A computer is problematic when it comes to high speed performance.')
    doc7 = NLP('A computer science course does not provide sufficient time for this kind of training in creative design, but it can provide the essential elements: an understanding of the user s needs, and an understanding of potential solutions.')
    doc6 = NLP('In professional work, the most important attributes for \
    HCI experts are to be both creative and practical, placing design at the centre of the field.')

    tr, = get_stanford_tree(doc2)
    tr.pretty_print()

    print(tr[0, 0])
    print(tr[0, 1])

    tr[0, 1], tr[0, 0] = tr[0, 0], tr[0, 1]
    print(tr)

    tr.pretty_print()
    # sentences = [sent for sent in doc.sents]
    #
    # initialize_matchers()
    #
    # apo = NLP('The magical Sam, my crazy brother from Ohio, eats red meat.')
    # subordinate = NLP('As John slept, I studied.')
    # subordinate_2 = NLP('The accident happened as the night was falling.')

    # APPOS.add("Appositive", None, [{DEP: 'appos'}])
    # matches = APPOS(apo)

    # extract_appositives(apo)
    # extract_subordinate(subordinate_2)

    # subordinate.print_tree()
    # for tok in subordinate:
    #     print(tok.text, tok.dep_)

    # displacy.serve(doc7, style='dep', port=5001)

    # tr, = get_stanford_tree(apo)
    # tr.pretty_print()

    # print(find_root(apo))
    # print_noun_chunks(apo)

    # for ent_id, start, end in matches:
    #     span = apo[start:end]
    #     # First token is our noun_phrase_0
    #     np_0 = span[0]
    #     print(np_0)
    #     for child in np_0.subtree:
    #         print(child)

    # displacy.serve(apo, style='dep', port=5001)
    # show_tree(sentences[0])

# def showTree(sent):
#     def __showTree(token, level):
#         tab = "\t" * level
#         sys.stdout.write("\n%s{" % (tab))
#         [__showTree(t, level+1) for t in token.lefts]
#         sys.stdout.write("\n%s\t%s [%s] (%s)" % (tab, token, token.dep_, token.tag_))
#         [__showTree(t, level+1) for t in token.rights]
#         sys.stdout.write("\n%s}" % (tab))
#     return __showTree(sent.root, 1)
