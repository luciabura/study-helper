import os
from queue import Queue

from nltk.parse.stanford import StanfordParser
from spacy import displacy
from spacy.matcher import Matcher

from neuralcoref import Coref
from question_generation import *
from text_processing.grammar import extract_noun_phrase, is_valid_sentence, find_parent_verb, \
    get_verb_correct_tense, remove_spans, get_subtree_span, safe_join
from utilities import NLP

# __location__ = os.path.realpath(
#     os.path.join(os.getcwd(), os.path.dirname(__file__)))

# parser_path = os.path.join(__location__, "stanford_parser/englishPCFG.ser.gz")
# jar_path = os.path.join(__location__, "stanford_parser/stanford-parser.jar")
# model_path = os.path.join(__location__, "stanford_parser/stanford-parser-models.jar")

# PARSER = StanfordParser(model_path=parser_path, path_to_jar=jar_path, path_to_models_jar=model_path)
from utilities.read_write import read_file

REL_PRONS_REM = ['which', 'who']
REL_PRON_ADD = ['where', 'when', 'what']

NLP.vocab["'"].is_alpha = True
NLP.vocab['"'].is_punct = True

MATCHER = Matcher(NLP.vocab)


def initialize_matcher_patterns():
    appositive = [{DEP: 'appos'}]
    conjoined_sentences_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN,
                             {POS: 'VERB', DEP: 'conj'}]
    conjoined_sentences_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN,
                             {POS: 'VERB', DEP: 'conj'}]

    # conjoined_subjects_1 = [{DEP: 'nsubj'}, ANY, {DEP: 'cc'}, ANY, {DEP: 'conj'}, ANY, {POS: 'VERB', DEP: 'ROOT'}]
    # conjoined_subjects_2 = [{DEP: 'nsubjpass'}, ANY, {DEP: 'cc'}, ANY, {DEP: 'conj'}, ANY, {POS: 'VERB', DEP: 'ROOT'}]

    commas = [{ORTH: ','}, ANY_TOKEN, {ORTH: ','}]
    comma_end = [{ORTH: ','}, ANY_TOKEN, {ORTH: '.'}]
    parenthesis = [{ORTH: '('}, ANY_TOKEN, {ORTH: ')'}]

    adjectival_modifier = [{DEP: 'acl'}]
    relative_clause_modifier = [{DEP: 'relcl'}]

    clausal_complement = [{DEP: 'ccomp'}]

    leading_pp = [{DEP: 'prep'}, ANY_TOKEN, {DEP: 'pobj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {TAG: '.'}]

    adverbial_clause_modifier = [{DEP: 'advcl'}]

    # NEW PATTERN: noun, relcl, verb the story which was incredible was told by ... bla bla

    # test_pattern = [{DEP: 'nsubj'}, ANY_ALPHA, {POS: 'VERB'}, ANY_TOKEN, {DEP: 'nsubj'}]

    MATCHER.add("CONJ_SENT", None, conjoined_sentences_1)
    MATCHER.add("CONJ_SENT", None, conjoined_sentences_2)
    MATCHER.add("PUNCT", None, commas)
    MATCHER.add("PUNCT", None, comma_end)
    MATCHER.add("PUNCT", None, parenthesis)
    MATCHER.add("ACL", None, adjectival_modifier)
    MATCHER.add("RELCL", None, relative_clause_modifier)
    MATCHER.add("APPOS", None, appositive)
    MATCHER.add("CCOMP", None, clausal_complement)
    MATCHER.add("LEAD_PP", None, leading_pp)
    MATCHER.add("SUBORD", None, adverbial_clause_modifier)

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


def extract_subordinate(match):
    advcl = match.get_token_by_attributes(dependency='advcl')
    advcl_np = extract_noun_phrase(advcl, match.sentence)

    parent_verb = find_parent_verb(advcl)
    subject = None

    if parent_verb is None:
        return []

    for tok in parent_verb.subtree:
        if tok.dep_.startswith("nsubj"):
            subject = tok

    if subject is None:
        return []

    subject_np = extract_noun_phrase(subject, match.sentence)

    verb = get_verb_correct_tense(dependant_noun_phrase=subject_np,
                                  dependant_verb=parent_verb, verb_lemma='be')

    sentence = [tok.text for tok in subject_np]
    sentence.extend([verb])
    sentence.extend([tok.text for tok in advcl_np])
    sentence.extend(['.'])

    return [sentence]


def extract_conjugate_sentences(match):
    sentences = []
    sentence = match.sentence
    root_verb = match.span.root
    conj = match.get_token_by_attributes(dependency='cc', head=root_verb)

    if conj is None:
        return []

    conj_root_verb = sentence[(conj.i + 1):].root

    if conj_root_verb.head != root_verb:
        print(conj_root_verb, root_verb)
        return []

    # Could well happen that there is an issue here with eg: Anna did this and also did this.
    # In the above case, the second this will not identify its subject...
    conj_subtree = get_subtree_span(conj_root_verb, sentence)

    # conj_sent = NLP(' '.join([tok.text for tok in conj_subtree]))
    conj_sent = [tok.text for tok in conj_subtree]

    root_sent = remove_spans(sentence, [conj_subtree, [conj]])
    if root_sent[-1] in [",", ";"]:
        root_sent.pop()
    root_sent.append('.')

    sentences.append(conj_sent)
    sentences.append(root_sent)

    return sentences


def appositive_sentence(appos, sentence):
    dependant = appos.head

    # Add check if dependant is verb, we ideally don't want to make a sentence...
    # This happens due to mistakenly tagging the dependencies
    if dependant.pos_ == "VERB":
        return None, None

    appositive_np = extract_noun_phrase(appos, sentence)
    dependant_np = extract_noun_phrase(dependant, sentence,
                                       exclude_span=appositive_np, discard_punct=[",", ":", "(", ")"])

    parent_verb = find_parent_verb(dependant)

    if parent_verb is None:
        # X: the study of Y.
        verb = 'is'
    else:
        verb = get_verb_correct_tense(dependant_noun_phrase=dependant_np,
                                      dependant_verb=parent_verb, verb_lemma='be')

    # Need dependant noun phrase to construct sentence
    appositive_sent = [tok.text for tok in dependant_np if tok.pos_ != "PUNCT"]
    appositive_sent.append(verb)
    appositive_sent.extend([tok.text for tok in appositive_np if tok.pos_ != "PUNCT"])

    return appositive_sent, appositive_np


def extract_appositives(match):
    sentences = []
    sentence = match.sentence

    appos = match.get_token_by_attributes(dependency='appos')

    appositive_sent, appositive_np = appositive_sentence(appos, sentence)
    if appositive_sent is None:
        return []

    preceding_token_index = appositive_np[0].i - 1
    next_token_index = appositive_np[-1].i + 1

    if sentence[preceding_token_index].pos_ == "PUNCT":
        sentence = remove_spans(sentence, [sentence[preceding_token_index : next_token_index + 1]])
    else:
        sentence = remove_spans(sentence, [appositive_np])

    sentences.append(sentence)
    sentences.append(appositive_sent)

    return sentences


def extract_adjectival_modifier(match):
    _sentences = []
    sentence = match.sentence
    adj_mod = match.get_first_token()
    subject = adj_mod.head

    if subject is None:
        return []

    adj_mod_np = extract_noun_phrase(adj_mod, sentence)
    subject_np = extract_noun_phrase(subject, sentence, exclude_span=adj_mod_np, discard_punct=[','])

    verb = get_verb_correct_tense(dependant_noun_phrase=subject_np, dependant_verb=adj_mod, verb_lemma='be')

    adj_sent = [tok.text for tok in subject_np]
    adj_sent.extend([verb])
    adj_sent.extend([tok.text for tok in adj_mod_np])

    # adj_sent = NLP(' '.join(adj_sent))
    _sentences.append(adj_sent)

    return _sentences


def extract_from_punct(match):
    sents = []

    # avoiding the punctuation marks
    punct_span = match.sentence[match.get_first_token().i + 1: match.get_last_token().i]
    first_after_punct = punct_span[0]

    sentence = remove_spans(match.sentence, spans=[match.span])

    punct_sent = []
    # Ensure relative clause is treated, may move this to separate pattern match
    # for token in punct_span:
    #     if token.dep_ == 'relcl':
    #         subject_np = extract_noun_phrase(token.head, match.sentence,
    #                                          discard_commas=True, exclude_span=match.span)
    #         if subject_np is None:
    #             continue
    #
    #         punct_sent.extend([tok.text for tok in subject_np])
    #         # Remove the which, who and add is/were etc for where, when..
    #         if first_after_punct.text in REL_PRON_ADD:
    #             verb = get_verb_correct_tense(subject_np, token)
    #             punct_sent.append(verb)
    #
    #         punct_sent.extend([tok.text for tok in punct_span] + ['.'])
    #
    #         if first_after_punct.text in REL_PRONS_REM:
    #             punct_sent.remove(first_after_punct.text)

    if len(punct_sent) == 0:
        punct_sent = [tok.text for tok in punct_span] + ['.']

    sents.append(punct_sent)
    sents.append(sentence)

    return sents


def TP(match):
    print(match.span)
    return []


def extract_relative_clause_modifier(match):
    sentences = []
    verb_relcl = match.get_last_token()
    noun = verb_relcl.head

    relcl_span = extract_noun_phrase(verb_relcl, match.sentence)
    noun_phrase = extract_noun_phrase(noun, match.sentence,
                                      discard_punct=[","], exclude_span=relcl_span)

    relcl_sent = []
    relcl_sent.extend([tok.text for tok in noun_phrase]) # if tok.pos_ != 'PUNCT'])

    first_tok = relcl_span[0]
    if first_tok.tag_ == 'WP$':
        relcl_sent.append("'s")
    elif first_tok.tag_ == 'WDT':
        pass
    else:
        verb = get_verb_correct_tense(noun_phrase, verb_relcl, verb_lemma='be')
        relcl_sent.append(verb)
        relcl_sent.append(first_tok.text)

    relcl_sent.extend([tok.text for tok in relcl_span[1:]])
    remaining_sent = remove_spans(match.sentence, spans=[relcl_span])

    sentences.extend([relcl_sent, remaining_sent])

    return sentences


def extract_from_clausal_complement(match):
    sentences = []
    ccomp_span = get_subtree_span(match.get_token_by_attributes(dependency="ccomp"),sentence=match.sentence)
    ccomp_mark = [tok for tok in ccomp_span if tok.dep_ == "mark"]

    if ccomp_mark:
        ccomp_sent = [tok for tok in remove_spans(ccomp_span, spans=[ccomp_mark])]
    else:
        ccomp_sent = [tok.text for tok in ccomp_span]

    sentences.extend([ccomp_sent])

    return sentences


def extract_leading_pp(match):
    root_verb = match.get_token_by_attributes(dependency='ROOT')
    preposition = match.get_first_token()
    pp_span = get_subtree_span(preposition, match.sentence)
    sent_no_pp_span = extract_noun_phrase(root_verb, match.sentence, exclude_span=pp_span)

    sentence = [tok.text for tok in sent_no_pp_span]
    sentence = sentence[:-1]
    sentence.extend([tok.lower_ for tok in pp_span])
    sentence.append('.')

    return [sentence]


pattern_to_simplification = {
    "APPOS": lambda match: extract_appositives(match),
    "SUBORD": lambda match: extract_subordinate(match),
    "CONJ_SENT": lambda match: extract_conjugate_sentences(match),
    "CC_SUBJ": lambda match: extract_conjoined_subjects(match),
    "ACL": lambda match: extract_adjectival_modifier(match),
    "RELCL": lambda match: extract_relative_clause_modifier(match),
    "PUNCT": lambda match: extract_from_punct(match),
    "CCOMP": lambda match: extract_from_clausal_complement(match),
    "LEAD_PP": lambda match: extract_leading_pp(match),
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
    t13 = NLP(
        u'One interesting aspect of word2vec training is the use of negative sampling instead of softmax (which is computationally very expensive)')
    t14 = NLP(u'The heavy rain, which was unusual for that time of year, destroyed most of the plants in my garden.')
    t15 = NLP(u'The rule derived by Andrej can be used freely nowadays.')
    t16 = NLP(u'They experimented on the people fired last year.')
    t17 = NLP(u'They kept thinking about a way to escape.')
    t18 = NLP(u'He came up with the idea of a time machine.')
    t19 = NLP(u"They were looking forward to finishing the lesson on RISC architecture.")
    t20 = NLP(u"He runs, but he is too slow for them.")
    t21 = NLP(
        u"A computer science course does not provide sufficient time for this kind of training in creative design, but it can provide the essential elements: an understanding of the user’s needs, and an understanding of potential solutions. ")
    t22 = NLP(u"Compositional semantics is the construction of meaning (often expressed as logic) based on syntax.")
    t23 = NLP(u"The computer whose hard disk was broken was taken away.")
    t24 = NLP(u"The book, which is now at the store, has sold over 1000 copies so far.")
    t25 = NLP(u"Apple’s first logo, designed by Jobs and Wayne, depicts Sir Isaac Newton sitting under an apple tree.")
    t26 = NLP(u"John, her brother, is going to visit us.")
    t26 = NLP(u"As John slept, she cried.")
    t26 = NLP(u"They marched ahead although they were told to stay put.")
    t26 = NLP(u"This includes the Long short term memory (LSTM) models which are a development of basic RNNs, which have been found to be more effective for at least some language applications.")
    t26 = NLP("She decided she did not want any more tea, so shook her head when the waiter reappeared.")
    t26 = NLP("In January, Ann stopped wearing her winter coat.")
    t26 = NLP("Le and Mikolov (2014) describe doc2vec, which is a modification of word2vec.")
    t26 = NLP("The twins, hoping to get a good grade, studied.")
    t26 = NLP("Mara had her car stolen.")
    # t26 = NLP("Ann's computer was turned off.")
    # up is dep_ = 'prt'

    # sent, appositive = extract_appositives(t2)
    # print(sent)
    # print(appositive)

    show_dependencies(t26)
    # print([(tok.dep_, tok.pos_) for tok in t10])
    # for s in simplify_sentence(t25.text):
    #     print(s)

    # coref(t11)3


def make_spacy_sentence(text_list):
    text = safe_join(text_list)
    doc = NLP(text)
    return doc


def is_simple(sent):
    if any(tok.text in [',', '(', '{', '+'] for tok in sent):
        return False

    return True


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

        if sent not in sentences_ and is_simple(sent):
            sentences_.append(sent)

        for ent_id, start, end in matches:
            match = Match(ent_id, start, end, sent)
            pattern_name = NLP.vocab.strings[ent_id]

            sentence_components = list(filter(lambda ss: ss not in seen_sents,
                                              handle_match(pattern_name)(match)))
            seen_sents.extend(sentence_components)

            simplified_sentences = list(map(make_spacy_sentence, sentence_components))

            valid_sentences = list(filter(is_valid_sentence, simplified_sentences))
            # special check if contains punctuation maybe?
            for s in valid_sentences:
                # sentences_.append(s)
                queue.put(s)

    valid_sentences = list(filter(is_valid_sentence, sentences_))
    return valid_sentences


def get_coreferences(text):
        # text = u"John and Mary paint. They both like pie."
        # text = u"A computer science course does not provide sufficient time for this kind of training in creative design, but it can provide the essential elements an understanding of the user s needs, and an understanding of potential solutions."
        # text = u"My sister has a dog. She loves that dog."
        # text = u"Apple's new iPhone was an incredible hit. Its price was, on the other hand, extremely unwieldy."

        coref = Coref(nlp=NLP)

        # clusters = coref.one_shot_coref(utterances=u"My sister has a dog. She loves that dog.")
        clusters = coref.one_shot_coref(utterances=text)
        # print(clusters)

        # mentions = coref.get_mentions()
        # print(mentions)

        # utterances = coref.get_utterances()
        # print(utterances)

        resolved_utterance_text = coref.get_resolved_utterances()
        print(resolved_utterance_text)

        coreferences = coref.get_most_representative()
        print(coreferences)

        return resolved_utterance_text


def show_dependencies(sentence, port=5001):
    displacy.serve(sentence, style='dep', port=port)


def simplify_sent_test():
    sentence = input("Sentence to simplify:")
    sents = simplify_sentence(sentence)
    for s in sents:
        print(s)


def resolve_coreferences():
    FP = input('Filepath:')
    text = read_file(FP)
    get_coreferences(text)


if __name__ == '__main__':
    doc = NLP(u'The set of natural numbers is countably infinite and different from the compatibility of systems.')
    # for tok in doc:
    #     print (tok.text, tok.dep_, tok.head)
    # show_dependencies(doc)
    # get_coreferences()

    sentences()
    # simplify_sent_test()

    # resolve_coreferences()