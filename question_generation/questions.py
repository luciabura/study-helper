import math
from collections import OrderedDict

import spacy
from spacy import displacy
from spacy.attrs import DEP, POS, IS_ALPHA
from spacy.matcher import Matcher

from keyword_extraction.keywords_filtered import get_keywords_with_scores
from preprocessing import preprocessing as preprocess
from summarization.summary import get_sentences_with_keywords_and_scores

NLP = spacy.load('en_core_web_md')
INFINITY = math.inf

ATTR = Matcher(NLP.vocab)
DOBJ = Matcher(NLP.vocab)
AGENT = Matcher(NLP.vocab)
POBJ = Matcher(NLP.vocab)

MATCHER = Matcher(NLP.vocab)

OP = 'OP'
ANY = {IS_ALPHA: True, OP: "*"}

WHO_ENTS = ['PERSON', 'NORP']
WHEN_ENTS = ['DATE']
WHERE_ENTS = ['LOCATION', 'FACILITY', 'ORG', 'LOC', 'GPE']
HOW_MUCH_ENTS = ['MONEY', 'PERCENT']
WHEN_PREPS = ['before', 'after', 'since', 'until']
WHERE_PREPS = ['to', 'on', 'at', 'over', 'in', 'behind', 'above', 'below', 'from', 'inside', 'outside']


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


def choose_wh_word(span):
    """
        Takes a span of tokens and returns the corresponding wh-word to construct a sentence with it
        could be a word, could be a phrase
        eg: England: subj -> Which location
        Ann: subj -> who?
    """
    wh_word = 'What'
    # Choose 'how much'
    if any(tok.ent_type_ in HOW_MUCH_ENTS for tok in span):
        return 'How much'

    # Choose 'who'
    if any(tok.ent_type_ in WHO_ENTS for tok in span) or has_pronouns(span):
        return 'Who'

    # Choose 'when'
    if any(tok.ent_type_ in WHEN_ENTS or tok.text in WHEN_PREPS for tok in span):
        return 'When'

    # Choose 'where'
    for tok in span:
        if tok.ent_type_ in WHERE_ENTS and tok.head.dep_ == 'prep' and tok.head.text in WHERE_PREPS:
            return 'Where'

    return wh_word


def initialize_patterns():
    # This will match wrongly on is created, was made etc
    attribute_1 = [{DEP: 'nsubj'}, ANY, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'attr'}]
    attribute_2 = [{DEP: 'nsubjpass'}, ANY, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'attr'}]

    direct_object_1 = [{DEP: 'nsubj'}, ANY, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'dobj'}]
    direct_object_2 = [{DEP: 'nsubjpass'}, ANY, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'dobj'}]

    prep_object_1 = [{DEP: 'nsubj'}, ANY, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'prep'},
                     ANY, {DEP: 'pobj'}]
    prep_object_2 = [{DEP: 'nsubjpass'}, ANY, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'prep'},
                     ANY, {DEP: 'pobj'}]

    agent_1 = [{DEP: 'nsubj'}, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'agent'}, ANY, {DEP: 'pobj'}]
    agent_2 = [{DEP: 'nsubjpass'}, {POS: 'VERB', DEP: 'ROOT'}, ANY, {DEP: 'agent'}, ANY, {DEP: 'pobj'}]

    MATCHER.add("ATTR", None, attribute_1)
    MATCHER.add("ATTR", None, attribute_2)
    MATCHER.add("DOBJ", None, direct_object_1)
    MATCHER.add("DOBJ", None, direct_object_2)
    MATCHER.add("POBJ", None, prep_object_1)
    MATCHER.add("POBJ", None, prep_object_2)
    MATCHER.add("AGENT", None, agent_1)
    MATCHER.add("AGENT", None, agent_2)


def sort_scores(scores):
    sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: t[1], reverse=True))

    return sorted_scores


def is_3rd_person(verb):
    return verb.tag_ == 'VBZ'


def prepare_question_verb(verb, sentence, includes_subject):
    # Have to check verb for more than 1 word => needn't prepare question verb
    # eg: can take, will see, etc
    vp = get_verb_phrase(verb, sentence)
    q_verb = []

    if len(vp) > 1:
        # This means we already have a composite verb
        q_verb = []
        # Should find better way to treat special cases
        if not includes_subject and (vp[0].text == 'am' or vp[0].text == 'are'):
            q_verb.append('is')
        elif not includes_subject and vp[0].text == 'were':
            q_verb.append('was')
        elif not includes_subject and vp[0].text == 'have':
            q_verb.append('has')
        else:
            q_verb.append(vp[0].text)

        q_verb.extend([tok.text for tok in vp[1:]])
        return q_verb

    if is_past_tense(verb):
        if includes_subject and verb is not 'was':
            # Who did Harry meet? from Harry met Sally.
            q_verb.append('did')
            q_verb.append(verb.lemma_)
        else:
            # Who met Harry? from Harry met Sally.
            q_verb.append(verb.text)

    else:
        if is_3rd_person(verb):
            if includes_subject and verb.text != 'is':
                # Where does Harry go? from Harry goes to the store.
                q_verb.append('does')
                q_verb.append(verb.lemma_)
            else:
                # Who goes to the store? from Harry goes to the store
                q_verb.append(verb.text)
        else:
            q_verb.append(verb.text)

    return q_verb


def is_valid_subject(subject):
    """If it has any coreference or if it contains something like those, this etc, then it isn't"""
    return True


def format_phrase(span):
    """Decapitalize starting if it's not a proper noun"""
    phrase = []
    for tok in span:
        if tok.pos_ != 'PROPN':
            phrase.append(tok.lower_)
        else:
            phrase.append(tok.text)

    return ' '.join(phrase)


def generate_prepositional_questions(match):
    questions = []
    sentence = match.get_sentence()
    subject = match.get_first_token()
    pobject = match.get_last_token()
    verb = subject.head
    preposition = pobject.head

    # Add checks for wrong matches
    if preposition.head != verb:
        return

    subject_phrase = extract_noun_phrase(subject, sentence)
    pobject_phrase = extract_noun_phrase(preposition, sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_pobject = choose_wh_word(pobject_phrase)

    verb_form_with_subject = prepare_question_verb(verb, sentence, includes_subject=True)
    verb_form_with_pobject = prepare_question_verb(verb, sentence, includes_subject=False)

    if is_valid_subject(subject_phrase):
        question = [wh_word_pobject]

        if len(verb_form_with_subject) == 1:
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
        else:
            # I don't like this ... could have more than 2 in verb phrase, not in attribute, but other ones yes
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
            question.extend(verb_form_with_subject[1:])

        # Very bad fix
        if wh_word_pobject == 'What':
            question.append(preposition.text)

        question = ' '.join(question)

        question += '?'
        questions.append(question)

    question = [wh_word_subject]
    question.extend(verb_form_with_pobject)
    question.append(format_phrase(pobject_phrase))

    question = ' '.join(question)
    question += '?'
    questions.append(question)

    return questions


def generate_agent_questions(match):
    """Only matches the by agent preposition"""
    questions = []
    sentence = match.get_sentence()
    subject = match.get_first_token()
    pobject = match.get_last_token()
    verb = subject.head

    subject_phrase = extract_noun_phrase(subject, sentence)
    pobject_phrase = extract_noun_phrase(pobject, sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_pobject = choose_wh_word(pobject_phrase)

    verb_form_with_subject = prepare_question_verb(verb, sentence, includes_subject=True) + ['by']
    verb_form_with_pobject = prepare_question_verb(verb, sentence, includes_subject=False) + ['by']

    if is_valid_subject(subject_phrase):
        question = [wh_word_pobject]

        if len(verb_form_with_subject) == 1:
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
        else:
            # I don't like this ... could have more than 2 in verb phrase, not in attribute, but other ones yes
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
            question.extend(verb_form_with_subject[1:])

        question = ' '.join(question)
        question += '?'
        questions.append(question)

    question = [wh_word_subject]
    question.extend(verb_form_with_pobject)
    question.append(format_phrase(pobject_phrase))

    question = ' '.join(question)
    question += '?'
    questions.append(question)

    return questions


# def make_questions(sentence, wh_word, subj, verb, obj, preposition=None):
#     questions = []
#     verb_form_with_subject = prepare_question_verb(verb, sentence, includes_subject=True)
#     verb_form_with_pobject = prepare_question_verb(verb, sentence, includes_subject=False)
#
#     if is_valid_subject(subj):
#         question = [wh_word_pobject]
#
#         if len(verb_form_with_subject) == 1:
#             question.append(verb_form_with_subject[0])
#             question.append(format_phrase(subj))
#         else:
#             # I don't like this ... could have more than 2 in verb phrase, not in attribute, but other ones yes
#             question.append(verb_form_with_subject[0])
#             question.append(format_phrase(subj))
#             question.extend(verb_form_with_subject[1:])
#
#         question = ' '.join(question)
#         question += '?'
#         questions.append(question)
#
#     question = [wh_word_subject]
#     question.extend(verb_form_with_pobject)
#     question.append(format_phrase(pobject_phrase))
#
#     question = ' '.join(question)
#     question += '?'
#     questions.append(question)

def generate_attribute_questions(match):
    """Gets a match object for the attribute pattern. Returns a list of questions"""
    questions = []
    sentence = match.get_sentence()
    subject = match.get_first_token()
    verb = subject.head

    subject_phrase = extract_noun_phrase(subject, sentence)
    attribute = extract_noun_phrase(match.get_last_token(), sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_attribute = choose_wh_word(attribute)

    verb_form_with_subject = prepare_question_verb(verb, sentence, includes_subject=True)
    verb_form_with_attribute = prepare_question_verb(verb, sentence, includes_subject=False)

    if is_valid_subject(subject_phrase):
        questions.append('Describe or define ' + format_phrase(subject_phrase) + '.')
        question = [wh_word_attribute]

        if len(verb_form_with_subject) == 1:
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
        else:
            # I don't like this ... could have more than 2 in verb phrase, not in attribute, but other ones yes
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
            question.extend(verb_form_with_subject[1:])

        question = ' '.join(question)
        question += '?'
        questions.append(question)

    question = [wh_word_subject]
    question.extend(verb_form_with_attribute)
    question.append(format_phrase(attribute))

    question = ' '.join(question)
    question += '?'
    questions.append(question)

    return questions


def generate_dobj_questions(match):
    questions = []
    sentence = match.get_sentence()
    direct_object = match.get_last_token()
    subject = match.get_first_token()
    verb = subject.head

    # Means we've matched in error
    if direct_object.head != verb:
        return

    subject_phrase = extract_noun_phrase(subject, sentence)
    direct_object_phrase = extract_noun_phrase(direct_object, sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_object = choose_wh_word(direct_object_phrase)

    verb_form_with_subject = prepare_question_verb(verb, sentence, includes_subject=True)
    verb_form_with_object = prepare_question_verb(verb, sentence, includes_subject=False)

    if is_valid_subject(subject_phrase):
        questions.append('Describe the relation or interaction between '
                         + format_phrase(subject_phrase)
                         + ' and '
                         + format_phrase(direct_object_phrase)
                         + '.')
        question = [wh_word_object]

        if len(verb_form_with_subject) == 1:
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
        else:
            # I don't like this ... could have more than 2 in verb phrase, not in attribute, but other ones yes
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
            question.extend(verb_form_with_subject[1:])

        question = ' '.join(question)
        question += '?'
        questions.append(question)

    question = [wh_word_subject]

    question.extend(verb_form_with_object)
    question.append(format_phrase(direct_object_phrase))

    question = ' '.join(question)
    question += '?'
    questions.append(question)

    return questions


def generate_questions(text):
    # The protocol for getting sentences with corresponding scores and keywords
    tokens = preprocess.clean_and_tokenize(text)
    keywords_with_scores = get_keywords_with_scores(tokens)
    sentences = preprocess.sentence_tokenize(text)
    sentences_with_keywords_and_scores = get_sentences_with_keywords_and_scores(sentences, keywords_with_scores)

    sorted_sentences = list(sort_scores(sentences_with_keywords_and_scores))
    i = 0
    for sentence in sorted_sentences:
        # for token in sentence:
        # print(token.text, token.dep_)
        # showTree(sentence)
        print(sentence[0:2], i)
        i += 1
        # print(sentences_with_keywords_and_scores[sentence][1])
        # break


def has_pronouns(span):
    for word in span:
        if word.tag_ == 'PRP' or word.tag_ == 'PRP$':
            return True

    return False


def get_coreference(pronoun):
    """Implement if time -> pronoun get document coreference with neuralcoref"""
    return 0


def show_dependencies(sentence, port=5000):
    displacy.serve(sentence, style='dep', port=port)


def extract_noun_phrase(token, sentence, exclude_span=None):
    start_index = INFINITY
    end_index = -1

    for child in token.subtree:
        """This will fail in some cases, might want to try to just get full subtree, but then need to pay attention 
        what we call it on. For now, I'm going to call it only on subjects and object so should be OK to get subtree"""
        if child in exclude_span:
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
    for child in token.subtree:
        if (child.dep_.startswith("aux") and child.head == token) or child == token:
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
    return 0


def trial_sentences():
    text = NLP(u'Computer architecture is a set of rules that describe the functionality of computer systems.')
    text_2 = NLP(u"Apple's logo was designed by Steve Jobs.")
    text_3 = NLP(u"Stacy went to see Johnny at the store.")
    doc2 = NLP('John studied, hoping to get a good grade.')
    doc3 = NLP(u'I bought the book that inspired Bob.')
    doc4 = NLP(u'John gave Mary the book.')
    doc5 = NLP(u'Machines for calculating fixed numerical tasks such as the abacus have existed since antiquity.')
    doc6 = NLP(u'I am going to meet her for lunch tomorrow.')
    doc7 = NLP(u'This area became [a prohibited zone.')
    doc8 = NLP(u'The handle should be attached before the mantle.')
    doc9 = NLP(u'The United States is a terrible place to go to.')
    doc10 = NLP(u'From outside to inside, the chip contains several layers of complex intertwined transistors.')
    text_4 = NLP(u'A router is a networking device that forwards data packets between computer networks and a house is a living space.')
    d = NLP(u'A user centred design process provides a professional approach to creating software with functionality that users need.')

    text = NLP(u"Apple's logo was designed by Steve Jobs in early december 2006 in front of the Empire State Building.")

    show_dependencies(text)
    # for nc in text.noun_chunks:
    #     print(nc)

    # print(get_phrase(text[7]))


def is_past_tense(token):
    return token.dep_ == 'VBD' or token.dep_ == 'VBN'


def generate_q():
    # text = NLP(u'Computer Science is the study of both practical and theoretical approaches to computers.')
    # text = NLP(u'A router is a networking device that forwards data packets between computer networks.')
    # text = NLP(u"Stacy went to see Johnny at the store.")
    # text = NLP(u'From outside to inside, the chip contains several layers of complex intertwined transistors.')
    # text = NLP(u'A user centred design process could provide a professional approach to creating software with functionality that users need.')
    # text = NLP(u"Apple's logo was designed by Steve Jobs in early december 2006.")
    # text = NLP(u'John gave Mary the book.')
    # text = NLP(u"Apple's logo was designed by Steve Jobs in early december 2006.")
    # text = NLP(u'A computer scientist specializes in the theory of computation.')
    # text = NLP(u'Machines for calculating fixed numerical tasks such as the abacus have existed since antiquity.')
    text = NLP(u'The ecclesiastical parish of Navenby was originally placed in the Longoboby Rural Deanery.')
    all_questions = []

    for tok in text:
        print(tok.ent_type_)
    matches = MATCHER(text)
    for ent_id, start, end in matches:
        match = Match(ent_id, start, end, text)
        pattern_name = NLP.vocab.strings[ent_id]
        # print(pattern_name)
        questions = handle_match(pattern_name)(match)
        if questions:
            all_questions.extend(questions)

    for question in all_questions:
        print(question)


# Switch statement, sort of
pattern_to_question = {
    "ATTR": lambda match: generate_attribute_questions(match),
    "POBJ": lambda match: generate_prepositional_questions(match),
    "DOBJ": lambda match: generate_dobj_questions(match),
    "AGENT": lambda match: generate_agent_questions(match)
}


def handle_match(pattern_name):
    return pattern_to_question[pattern_name]


if __name__ == '__main__':
    # generate_questions(TEST_TEXT)
    # doc = NLP(u'Computer Science is the study of both practical and theoretical approaches to computers. A computer scientist specializes in the theory of computation.')
    # sentences = list(doc.sents)
    # show_dependencies(sentences[1].as_doc(), port=5001)
    initialize_patterns()
    generate_q()
    # trial_sentences()

