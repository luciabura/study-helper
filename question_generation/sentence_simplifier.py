import re

import os
import spacy
import sys

from nltk import Tree
from nltk.parse.stanford import StanfordParser
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


def get_stanford_tree(sentence):
    tagged_sentence = [(token.text, token.tag_) for token in sentence]
    t = PARSER.tagged_parse(tagged_sentence)
    return t


def draw_syntax_tree(tree):
    for line in tree:
        for tree in line:
            print(tree)


if __name__ == '__main__':
    doc = NLP(u'John studied, hoping to get better grades.')

    sentences = [sent for sent in doc.sents]
    tr, = get_stanford_tree(sentences[0])
    # tr.pretty_print()
    tt = tr.pformat()
    # print(tt)

    reg = re.compile(r'ROOT=root << (VP !< VP < (/,/=comma $+ /[^`].*/=modifier))')
    mat = reg.search(tt)
    print(mat.group())



    # pattern = re.compile(r'(<JJ>)*(<NN>|<NNS>|<NNP>)+')

    # # show_tree(sentences[1])
    # print(sentences[1].subtree)
    # command = "John studied, hoping to get better grades."
    # command2 = "As John slept, I studied."
    # en_doc = NLP(u'' + command2)

    # [to_nltk_tree(sent.root).pretty_print() for sent in en_doc.sents]
