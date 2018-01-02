import re

import os
import spacy
import sys

from nltk import Tree
from nltk.parse.stanford import StanfordParser
from spacy import displacy
from spacy.attrs import DEP
from spacy.matcher import Matcher

from question_generation.questions import Match, extract_noun_phrase, get_verb_phrase

from neuralcoref import Coref
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

parser_path = os.path.join(__location__, "stanford_parser/englishPCFG.ser.gz")
jar_path = os.path.join(__location__, "stanford_parser/stanford-parser.jar")
model_path = os.path.join(__location__, "stanford_parser/stanford-parser-models.jar")

PARSER = StanfordParser(model_path=parser_path, path_to_jar=jar_path, path_to_models_jar=model_path)

NLP = spacy.load('en_core_web_md')
RE_SUBORDINATES = " SBAR [ > VP < IN | > S|SINV ]  " + \
                      " !< (IN < if|unless|that)" + \
                      " < (S=sub !< (VP < VBG)) " + \
                      " >S|SINV|VP "

# def extract_subordinates(sentence):
#     matcher = re.compile(RE_SUBORDINATES, re.UNICODE)
#     matcher.findall()
#     extracted = []
#     return extracted
#
#
# def filtered_chunks(doc, pattern):
#     for chunk in doc.noun_chunks:
#
#         signature = ''.join(['<%s>' % w.tag_ for w in chunk])
#         print(signature, chunk)
#         if pattern.match(signature) is not None:
#             yield chunk
#
#
# def get_simplified_sentences(sentence):
#     simplified = []
#     simplified.append(sentence)
#
#     simplified.extend(extract_subordinates(sentence))
#
#
# def show_tree(sent):
#     def __show_tree(token, level):
#         tab = "\t" * level
#         sys.stdout.write("\n%s{" % (tab))
#         [__show_tree(t, level + 1) for t in token.lefts]
#         sys.stdout.write("\n%s\t%s [%s] (%s)" % (tab, token, token.dep_, token.tag_))
#         [__show_tree(t, level + 1) for t in token.rights]
#         sys.stdout.write("\n%s}" % (tab))
#
#     return __show_tree(sent.root, 1)
#
#
# def print_pattern(sent):
#     for token in sent:
#         print(str(token.tag_) + ' | ')
#
#
# def tok_format(tok):
#     return "_".join([tok.orth_, tok.tag_, tok.dep_])
#
#
# def to_nltk_tree(node):
#     if node.n_lefts + node.n_rights > 0:
#         return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
#     else:
#         return tok_format(node)


def is_plural(token):
    return token.tag_ == 'NNS' or token.tag_ == 'NNPS'


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


def appositive_sentence(appos, sentence):
    dependant = appos.head
    appositive_np = extract_noun_phrase(appos, sentence)
    dependant_np = extract_noun_phrase(dependant, sentence, exclude_span=appositive_np)

    if any(is_plural(tok) for tok in dependant_np):
        verb = u'are'
    else:
        verb = u'is'

    # Need dependant noun phrase to construct sentence
    components = [tok.text for tok in dependant_np]
    components.append(verb)
    components.extend([tok.text for tok in appositive_np])

    s = ' '.join(components)
    s += '.'

    sentence = NLP(s)
    return sentence, appositive_np


NLP = spacy.load('en_core_web_md')
APPOS = Matcher(NLP.vocab)


def extract_appositives(sentence):
    initialize_matchers()
    matches = APPOS(sentence)

    if matches is None:
        return

    for ent_id, start, end in matches:
        match = Match(ent_id, start, end, sentence)
        appos = match.get_tokens_by_dependency('appos')

    appositive_sent, appositive_np = appositive_sentence(appos, sentence)
    print(appositive_sent)

    sentence = sentence.remove(appositive_np)
    return sentence, appositive_sent


def initialize_matchers():
    pattern = [{DEP: 'appos'}]
    APPOS.add("Appositive", None, pattern)


def draw_syntax_tree(tree):
    for line in tree:
        for tree in line:
            print(tree)


def sentences():
    t1 = NLP(u'In professional work, the most important attributes for HCI experts are to be both creative and practical, placing design at the centre of the field.')
    t2 = NLP('A computer science course does not provide sufficient time for this kind of training in creative design, but it can provide the essential elements: an understanding of the user s needs, and an understanding of potential solutions.')

    extract_appositives(t2)

def coref(sentence):
    coref = Coref()
    # clusters = coref.one_shot_coref(utterances=u"My sister has a dog. She loves that dog.")
    clusters = coref.one_shot_coref(utterances=sentence)
    print(clusters)

    mentions = coref.get_mentions()
    print(mentions)

    utterances = coref.get_utterances()
    print(utterances)

    resolved_utterance_text = coref.get_resolved_utterances()
    print(resolved_utterance_text)

    coreferences = coref.get_most_representative()
    print(coreferences)


def show_dependencies(sentence, port=5000):
    displacy.serve(sentence, style='dep', port=port)


if __name__ == '__main__':
    sentences()




    # doc = NLP(u'John studied, hoping to get better grades.')
    #
    # sentences = [sent for sent in doc.sents]
    # tr, = get_stanford_tree(sentences[0])
    # # tr.pretty_print()
    # tt = tr.pformat()
    # # print(tt)
    #
    # reg = re.compile(r'ROOT=root << (VP !< VP < (/,/=comma $+ /[^`].*/=modifier))')
    # mat = reg.search(tt)
    # print(mat.group())

    # pattern = re.compile(r'(<JJ>)*(<NN>|<NNS>|<NNP>)+')

    # # show_tree(sentences[1])
    # print(sentences[1].subtree)
    # command = "John studied, hoping to get better grades."
    # command2 = "As John slept, I studied."
    # en_doc = NLP(u'' + command2)

    # [to_nltk_tree(sent.root).pretty_print() for sent in en_doc.sents]
