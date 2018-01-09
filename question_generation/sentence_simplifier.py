import os
from queue import Queue

from nltk.parse.stanford import StanfordParser
from spacy import displacy
from spacy.matcher import Matcher

# from neuralcoref import Coref
from question_generation import *
from text_processing.grammar import extract_noun_phrase, is_valid_sentence, find_parent_verb, \
    get_verb_correct_tense, remove_spans, get_subtree_span
from utilities import NLP

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

parser_path = os.path.join(__location__, "stanford_parser/englishPCFG.ser.gz")
jar_path = os.path.join(__location__, "stanford_parser/stanford-parser.jar")
model_path = os.path.join(__location__, "stanford_parser/stanford-parser-models.jar")

PARSER = StanfordParser(model_path=parser_path, path_to_jar=jar_path, path_to_models_jar=model_path)

REL_PRONS_REM = ['which', 'who']
REL_PRON_ADD = ['where', 'when', 'what']

MATCHER = Matcher(NLP.vocab)


def initialize_matcher_patterns():
    # IS_SUBJ = NLP.vocab.add_flag(lambda tok: tok.dep_ in ['nsubj', 'nsubjpass'])

    appositive = [{DEP: 'appos'}]
    conjoined_sentences_1 = [{DEP: 'nsubj'}, ANY_ALPHA, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN,
                             {POS: 'VERB', DEP: 'conj'}]
    conjoined_sentences_2 = [{DEP: 'nsubjpass'}, ANY_ALPHA, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN,
                             {POS: 'VERB', DEP: 'conj'}]

    # conjoined_subjects_1 = [{DEP: 'nsubj'}, ANY, {DEP: 'cc'}, ANY, {DEP: 'conj'}, ANY, {POS: 'VERB', DEP: 'ROOT'}]
    # conjoined_subjects_2 = [{DEP: 'nsubjpass'}, ANY, {DEP: 'cc'}, ANY, {DEP: 'conj'}, ANY, {POS: 'VERB', DEP: 'ROOT'}]

    commas = [{ORTH: ','}, ANY_ALPHA, {ORTH: ','}]
    parenthesis = [{ORTH: '('}, ANY_ALPHA, {ORTH: ')'}]

    adjectival_modifier = [{POS: 'NOUN'}, {DEP: 'acl'}]

    # test_pattern = [{DEP: 'ROOT', POS: 'VERB'}, ANY, {DEP: 'conj', POS: 'VERB'}] #, ANY, {POS: 'VERB', DEP: 'conj'}]
    # test_pattern = [{DEP: 'nsubj'}, ANY, {DEP: 'conj'}]
    test_pattern = [{DEP: 'nsubj'}, ANY_ALPHA, {POS: 'VERB'}, ANY_TOKEN, {DEP: 'nsubj'}]

    MATCHER.add("APPOS", None, appositive)
    MATCHER.add("CONJ_SENT", None, conjoined_sentences_1)
    MATCHER.add("CONJ_SENT", None, conjoined_sentences_2)
    MATCHER.add("PUNCT", None, commas)
    MATCHER.add("PUNCT", None, parenthesis)
    MATCHER.add("ACL", None, adjectival_modifier)
    # MATCHER.add("TP", None, test_pattern)


def post_process(sentence):
    """
    TODO: add check for trailing punctuation
    TODO: check if subject is a pronoun
    """
    pass


def extract_conjoined_subjects(match):
    subj_1 = match.get_first_token()

    return []


def extract_subordinate(sentence):
    for tok in sentence:
        if tok.dep_ == 'advcl':
            for sub in tok.subtree:
                print(sub)


def extract_conjugate_sentences(match):
    sentences = []
    sentence = match.sentence
    root_verb = match.span.root
    conj = match.get_token_by_attributes(dependency='cc', head=root_verb)
    conj_root_verb = sentence[(conj.i + 1):].root

    if conj_root_verb.head != root_verb:
        print(conj_root_verb, root_verb)
        return sentence.string.strip()

    conj_subtree = get_subtree_span(conj_root_verb, sentence)

    # conj_sent = NLP(' '.join([tok.text for tok in conj_subtree]))
    conj_sent = [tok.text for tok in conj_subtree]
    root_sent = remove_spans(sentence, [conj_subtree, [conj]])

    sentences.append(conj_sent)
    sentences.append(root_sent)

    return sentences


def appositive_sentence(appos, sentence):
    dependant = appos.head
    appositive_np = extract_noun_phrase(appos, sentence)
    dependant_np = extract_noun_phrase(dependant, sentence, exclude_span=appositive_np)

    parent_verb = find_parent_verb(dependant)

    if parent_verb is None:
        # X: the study of Y.
        verb = 'is'
    else:
        verb = get_verb_correct_tense(dependant_noun_phrase=dependant_np, dependant_verb=parent_verb, verb_lemma='be')

    # Need dependant noun phrase to construct sentence
    appositive_sent = [tok.text for tok in dependant_np if tok.tag_ not in [':', ',']]
    appositive_sent.append(verb)
    appositive_sent.extend([tok.text for tok in appositive_np])

    # s = ' '.join(components)
    # s += '.'

    # sentence = NLP(s)

    return appositive_sent, appositive_np


def extract_appositives(match):
    sentences = []
    sentence = match.sentence

    appos = match.get_token_by_attributes(dependency='appos')

    appositive_sent, appositive_np = appositive_sentence(appos, sentence)

    sentence = remove_spans(sentence, [appositive_np])
    sentences.append(sentence)
    sentences.append(appositive_sent)

    return sentences


def extract_adjectival_modifier(match):
    _sentences = []
    sentence = match.sentence
    subject = match.get_first_token()
    adj_mod = match.get_last_token()

    adj_mod_np = extract_noun_phrase(adj_mod, sentence)
    subject_np = extract_noun_phrase(subject, sentence, exclude_span=adj_mod_np)

    verb = get_verb_correct_tense(dependant_noun_phrase=subject_np, dependant_verb=adj_mod, verb_lemma='be')

    adj_sent = [tok.text for tok in subject_np]
    adj_sent.extend([verb])
    adj_sent.extend([tok.text for tok in adj_mod_np])

    # adj_sent = NLP(' '.join(adj_sent))
    _sentences.append(adj_sent)

    return _sentences


def extract_from_punct(match):
    sents = []

    punct_span = match.sentence[match.get_first_token().i + 1: match.get_last_token().i]
    first_after_punct = punct_span[0]

    sentence = remove_spans(match.sentence, spans=[match.span])

    punct_sent = []
    # Ensure relative clause is treated, may move this to separate pattern match
    for token in punct_span:
        if token.dep_ == 'relcl':
            subject_np = extract_noun_phrase(token.head, match.sentence, exclude_span=match.span)
            if subject_np is None:
                continue

            punct_sent.extend([tok.text for tok in subject_np])
            # Remove the which, who and add is/were etc for where, when..
            if first_after_punct.text in REL_PRON_ADD:
                verb = get_verb_correct_tense(subject_np, token)
                punct_sent.append(verb)

            punct_sent.extend([tok.text for tok in punct_span] + ['.'])

            if first_after_punct.text in REL_PRONS_REM:
                punct_sent.remove(first_after_punct.text)

    if len(punct_sent) == 0:
        # punct_sent = NLP(' '.join([tok.text for tok in punct_span]) + '.')
        punct_sent = [tok.text for tok in punct_span] + ['.']
    # else:
        # punct_sent = NLP(' '.join(punct_sent))

    sents.append(punct_sent)
    sents.append(sentence)

    return sents


def TP(match):
    print(match.span)
    return []


pattern_to_simplification = {
    "APPOS": lambda match: extract_appositives(match),
    "SUBORD": lambda match: extract_subordinate(match),
    "CONJ_SENT": lambda match: extract_conjugate_sentences(match),
    "CC_SUBJ": lambda match: extract_conjoined_subjects(match),
    "PUNCT": lambda match: extract_from_punct(match),
    "ACL": lambda match: extract_adjectival_modifier(match),
    "TP": lambda match: TP(match)
}


def handle_match(pattern_name):
    return pattern_to_simplification[pattern_name]


def sentences():
    t1 = NLP(
        u'In professional work, the most important attributes for HCI experts are to be both creative and practical, placing design at the centre of the field.')
    t2 = NLP(
        u'A computer science course does not provide sufficient time for this kind of training in creative design, but it can provide the essential elements: an understanding of the user s needs, and an understanding of potential solutions.')
    t3 = NLP(u'A router is a forwarding device and it helps with inter-network communications.')
    t4 = NLP(u'The cloud and wireless technology have both been invented in the last decade.')
    t5 = NLP(u'Harry and Sally have never been to London.')
    t6 = NLP(
        u'A user centred design process, as taught in earlier years of the tripos and experienced in many group design projects, provides a professional approach to creating software with functionality that users need.')
    t7 = NLP(u'However, John studied, hoping to get a good grade.')
    t8 = NLP(u'That will not happen if the cork is placed at the right time.')
    t9 = NLP(u'As far as current studies go, this is the best available solution.')
    t10 = NLP(
        u'Architects and product designers need a thorough technical grasp of the materials they work with, but the success of their work depends on the creative application of this technical knowledge.')
    t11 = NLP(u'John writes and Mary paints. They both like pie.')
    t12 = NLP(u'In principle, the RBMs can be trained separately and then fine-tuned in combination.')
    t13 = NLP(u'One interesting aspect of word2vec training is the use of negative sampling instead of softmax (which is computationally very expensive)')
    t14 = NLP(u'The heavy rain, which was unusual for that time of year, destroyed most of the plants in my garden.')
    t15 = NLP(u'The rule derived by Andrej can be used freely nowadays.')
    t16 = NLP(u'They experimented on the people fired last year.')
    t17 = NLP(u'They kept thinking about a way to escape.')
    t18 = NLP(u'He came up with the idea of a time machine.')
    t19 = NLP(u"They were looking forward to finishing the lesson on RISC architecture.")
    t20 = NLP(u"He runs, but he is too slow for them.")
    t21 = NLP(u"A computer science course does not provide sufficient time for this kind of training in creative design, but it can provide the essential elements: an understanding of the user’s needs, and an understanding of potential solutions. ")


    # up is dep_ = 'prt'

    # sent, appositive = extract_appositives(t2)
    # print(sent)
    # print(appositive)

    # show_dependencies(t20)
    # print([(tok.dep_, tok.pos_) for tok in t10])
    for s in simplify_sentence(t21):
        print(s)

    # coref(t11)


def make_spacy_sentence(text_list):
    doc = NLP(' '.join(text_list))
    return doc


def simplify_sentence(sentence, coreferences={}):
    """The main idea is the following:
    1. Check is anything is in coreferences and resolve the sentence
    2. Run matcher against sentence and extract sentences appropriately -> consider case """

    initialize_matcher_patterns()
    sentences_ = []
    seen_sents = []

    queue = Queue()
    queue.put(NLP(sentence))
    while not queue.empty():
        sent = queue.get()
        matches = MATCHER(sent)

        if sent not in sentences_:
            sentences_.append(sent)

        for ent_id, start, end in matches:
            match = Match(ent_id, start, end, sent)
            pattern_name = NLP.vocab.strings[ent_id]

            print(pattern_name)
            # print(match.span)

            sentence_components = list(filter(lambda s: s not in seen_sents,
                                              handle_match(pattern_name)(match)))
            seen_sents.extend(sentence_components)

            simplified_sentences = list(map(make_spacy_sentence, sentence_components))

            for s in simplified_sentences:
                sentences_.append(s)
                queue.put(s)

    valid_sentences = list(filter(is_valid_sentence, sentences_))
    return valid_sentences


def get_coreferences(text):
    pass
    #     # text = u"John and Mary paint. They both like pie."
    #     # text = u"My sister has a dog. She loves that dog."
    #     coref = Coref(nlp=NLP)
    #
    #     # clusters = coref.one_shot_coref(utterances=u"My sister has a dog. She loves that dog.")
    #     clusters = coref.one_shot_coref(utterances=text)
    #     # print(clusters)
    #
    #     mentions = coref.get_mentions()
    #     print(mentions)
    #
    #     # utterances = coref.get_utterances()
    #     # print(utterances)
    #
    #     resolved_utterance_text = coref.get_resolved_utterances()
    #     print(resolved_utterance_text)
    #
    #     coreferences = coref.get_most_representative()
    #     print(coreferences)
    #
    #     return coreferences, resolved_utterance_text


def show_dependencies(sentence, port=5000):
    displacy.serve(sentence, style='dep', port=port)


if __name__ == '__main__':
    doc = NLP(u'The set of natural numbers is countably infinite and different from the compatibility of systems.')
    # for tok in doc:
    #     print (tok.text, tok.dep_, tok.head)
    # show_dependencies(doc)
    sentences()
