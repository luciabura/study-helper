import math
from collections import OrderedDict

from spacy.matcher import Matcher
from spacy.tokens import Token, doc
from wordfreq import word_frequency

import question_generation.sentence_simplifier as simplifier
# from evaluation.question_evaluation import spacy_perplexity
from keyword_extraction.keywords_TR_lem import KeyPhrase, Keyword
from neuralcoref import Coref
from question_generation import *
from summarization.sentence_provider import SentenceProvider, Sentence
# from summarization.summary import get_sentences_with_keywords_and_scores
from text_processing import preprocessing as preprocess
from text_processing.grammar import *
from utilities import NLP
from utilities.read_write import read_file

WHO_ENTS = ['PERSON', 'NORP']
WHEN_ENTS = ['DATE']
WHERE_ENTS = ['LOCATION', 'FACILITY', 'ORG', 'LOC', 'GPE']
HOW_MUCH_ENTS = ['MONEY', 'PERCENT']
WHEN_PREPS = ['before', 'after', 'since', 'until', 'when']
WHERE_PREPS = ['to', 'on', 'at', 'over', 'in', 'behind', 'above', 'below', 'from', 'inside', 'outside']

MATCHER = Matcher(NLP.vocab)


class Question:
    def __init__(self, question, sentence, answer):
        self.content = NLP(question)
        # Assumes we get a sentence object of the form described by Sentence in sentence_provider
        self.sentence = sentence
        self.answer = answer

        self.score = 0

        self.__compute_score()

    def __compute_score(self):
        # Exclude questions which are referring expressions
        # eg: What is he looking at?

        for tok in self.content:
            if (tok.tag_ == 'PRON' or tok.tag_ == 'PRP') or \
                    (tok.tag_ == 'DT' and tok.dep_.startswith('nsubj')):
                if tok.text in ['we', 'you']:
                    continue
                self.score = 0
                return

        # Calculating the score for keywords within Question
        score_q = 0
        div = 0
        alpha = 0.955

        for kp in self.sentence.key_phrases:
            if kp.text in self.content.text.lower():
                div += 1
                surprise_factor_normalized = sequence_surprize(kp.text)
                score_q += alpha * kp.score + (1 - alpha) * surprise_factor_normalized
        if div:
            score_q = score_q / div

        # Calculating the score for keywords within answer phrase
        text_answer = [tok.lower_ for tok in self.answer]
        text_answer = ' '.join(text_answer)

        score_kp = 0
        div = 0
        for kp in self.sentence.key_phrases:
            if kp.text in text_answer:
                div += 1
                surprise_factor_normalized = sequence_surprize(kp.text)
                score_kp += alpha * kp.score + (1 - alpha) * surprise_factor_normalized

        if div:
            score_kp = score_kp / div

        beta = 0.2
        if score_q and score_kp:
            score = (1.0 + beta ** 2) * (score_q * score_kp) / (score_q + (beta ** 2) * score_kp)
        elif score_q:
            score = score_q
        else:
            score = score_kp * (1 - beta)

        # if len(self.content) > 0:
        #     score /= math.log(len(self.content))

        self.score = score

    def similarity(self, question2):
        if isinstance(question2, Question):
            return self.content.similarity(question2.content)
        elif isinstance(question2, str):
            return self.content.similarity(NLP(question2))
        elif isinstance(question2, doc):
            return self.content.similarity(question2)
        else:
            return None


def sequence_surprize(text):
    word_list = text.split()
    av_s = 0
    for word in word_list:
        wf = word_frequency(word, lang='en') * 1e8
        if wf:
            av_s += 1 / wf
        else:
            av_s += 0.1

    if len(word_list) > 1:
        av_s /= math.log(len(word_list))
    return av_s


def initialize_question_patterns():
    # This will match wrongly on is created, was made etc
    attribute_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'attr'}]
    acomp_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'acomp'}]
    # attribute_1 = [{DEP: 'nsubj'}]
    attribute_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'attr'}]
    acomp_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'acomp'}]

    ccomp_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'ccomp'}]
    ccomp_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'ccomp'}]

    direct_object_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'}]
    direct_object_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'}]

    prep_object_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'prep'},
                     ANY_TOKEN, {DEP: 'pobj'}]
    prep_object_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'prep'},
                     ANY_TOKEN, {DEP: 'pobj'}]

    d_obj_p_obj = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'},
                   ANY_TOKEN, {DEP: 'pobj'}]
    d_obj_p_obj_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'},
                     ANY_TOKEN, {DEP: 'pobj'}]

    d_obj_p_obj_inv = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'pobj'},
                       ANY_TOKEN, {DEP: 'dobj'}]
    d_obj_p_obj_inv_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'pobj'},
                         ANY_TOKEN, {DEP: 'dobj'}]
    dative_pobj = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dative'},
                   ANY_TOKEN, {DEP: 'pobj'}]

    dative_object_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dative'},
                       ANY_TOKEN, {DEP: 'dobj'}]
    dative_object_2 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'},
                       ANY_TOKEN, {DEP: 'dative'}]

    agent_1 = [{DEP: 'nsubj'}, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'agent'}, ANY_TOKEN, {DEP: 'pobj'}]
    agent_2 = [{DEP: 'nsubjpass'}, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'agent'}, ANY_TOKEN, {DEP: 'pobj'}]

    MATCHER.add("ATTR", None, attribute_1)
    MATCHER.add("ATTR", None, attribute_2)
    MATCHER.add("ATTR", None, acomp_1)
    MATCHER.add("ATTR", None, acomp_2)
    MATCHER.add("DOBJ", None, direct_object_1)
    MATCHER.add("DOBJ", None, direct_object_2)
    MATCHER.add("POBJ", None, prep_object_1)
    MATCHER.add("POBJ", None, prep_object_2)
    MATCHER.add("AGENT", None, agent_1)
    MATCHER.add("AGENT", None, agent_2)
    MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj)
    MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj_2)
    MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj_inv)
    MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj_inv_2)
    MATCHER.add("DOBJ-POBJ", None, dative_pobj)
    MATCHER.add("CCOMP", None, ccomp_1)
    MATCHER.add("CCOMP", None, ccomp_2)


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
    # Not good: could have tokens there but need to be immediatelly related to verb...
    # Amelia took a present to Joanna. What did Amelia take to Joanna?
    if any(tok.ent_type_ in WHO_ENTS for tok in span) or has_pronouns(span):
        return 'Who'

    # Choose 'when'
    if any(tok.ent_type_ in WHEN_ENTS
           or tok.text in WHEN_PREPS
           or tok.head.text in WHEN_PREPS
           for tok in span):
        return 'When'

    # Choose 'where'
    for tok in span:
        if tok.ent_type_ in WHERE_ENTS \
                and tok.head.dep_ == 'prep' \
                and tok.head.text in WHERE_PREPS:
            return 'Where'

    return wh_word


def sort_scores(scores):
    sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: t[1], reverse=True))

    return sorted_scores


def conjugate_verb(verb):
    if verb.tag_ == "VBP":
        if verb.lower_.endswith(("sh", "ch", "tch", "x", "x", "ss", "o")):
            return verb.lower_ + "es"
        elif verb.lower_.endswith("y"):
            last_2_chars = verb.lower_[-2:]
            if not is_vowel(last_2_chars[0]):
                return verb.lower_[:-2] + "ies"
            else:
                return verb.lower_ + "s"
        else:
            return verb.lower_ + "s"
    else:
        return verb.lower_


def get_verb_form(verb, includes_subject=False):
    verb_components = []
    if verb is None:
        return

    if not isinstance(verb, Token):
        print('Wrong parameter given. Expected Token item.')
        return

    if "'" in verb.lower_:
        verb_components.append(correct_short_verb_form(verb))

    elif is_past_tense(verb):
        if includes_subject and verb.lower_ not in ['was', 'were']:
            # Who did Harry meet? from Harry met Sally.
            verb_components.append('did')
            verb_components.append(verb.lemma_)
        else:
            # Who met Harry? from Harry met Sally.
            verb_components.append(verb.lower_)

    else:
        if is_3rd_person(verb):
            if includes_subject and verb.lower_ != 'is':
                # Where does Harry go? from Harry goes to the store.
                verb_components.append('does')
                verb_components.append(verb.lemma_)
            else:
                # Who goes to the store? from Harry goes to the store
                verb_components.append(verb.lower_)
        else:
            if not includes_subject and verb.lower_ == 'are':
                verb_components.append('is')
            elif not includes_subject and verb.lower_ == 'have':
                verb_components.append('has')
            elif not includes_subject:
                verb_components.append(conjugate_verb(verb))
            else:
                if includes_subject and verb.lower_ != 'are':
                    verb_components.extend(['do', verb.lower_])
                else:
                    verb_components.append(verb.lower_)

    return verb_components


def correct_short_verb_form(token_verb):
    """n't -> not"""
    if "'" in token_verb.lower_:
        return token_verb.lemma_.lower()
    else:
        return token_verb.lower_


def prepare_question_verb(vp, includes_subject):
    # Have to check verb for more than 1 word => needn't prepare question verb
    # eg: can take, will see, etc

    if len(vp) == 1:
        return get_verb_form(vp[0], includes_subject)

    else:
        # This means we have a composite verb
        q_verb = []
        # Should find better way to treat special cases
        if not includes_subject and vp[0].lemma_.lower() == 'be':
            if is_past_tense(vp[0]):
                # They were not here -> Who was not here?
                q_verb.append('was')
            else:
                # They are not here -> Who is not here?
                q_verb.append('is')

        elif not includes_subject and vp[0].lemma_.lower() == 'have' and not is_past_tense(vp[0]):
            # X have seen Y. -> Who has seen Y? (includes_subject = false)
            q_verb.append('has')

        elif includes_subject and vp[0].dep_ == 'neg':
            q_verb.append(vp[0].lower_)
            for tok in vp[1:]:
                if tok.lemma_ == 'do' and tok.dep_ != 'ROOT':
                    q_verb.insert(0, tok.lower_)
                    q_verb.extend(list(map(correct_short_verb_form, [v for v in vp[1:] if v.i > tok.i])))
                    break

                elif tok.dep_ == 'ROOT':
                    # X never saw Y => X did never see Y => Who did X never see?
                    verb_with_helper = get_verb_form(tok, includes_subject=True)
                    q_verb.insert(0, verb_with_helper[0])
                    q_verb.append(verb_with_helper[1])

                else:
                    q_verb.append(correct_short_verb_form(tok))

            return q_verb

        elif is_past_tense(vp[0]):
            verb = get_verb_form(vp[0], includes_subject)
            q_verb = verb
        else:
            q_verb.append(correct_short_verb_form(vp[0]))

        q_verb.extend(list(map(correct_short_verb_form, [tok for tok in vp[1:]])))
        return q_verb


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
    sentence = match.sentence
    subject = match.get_first_token()
    pobject = match.get_last_token()
    verb = subject.head
    preposition = pobject.head

    # Add checks for wrong matches
    if preposition.head != verb:
        return

    subject_phrase = extract_noun_phrase(subject, sentence)
    pobject_phrase = extract_noun_phrase(preposition, sentence)
    verb_phrase = get_verb_phrase(verb, sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_pobject = choose_wh_word(pobject_phrase)

    verb_form_with_subject = prepare_question_verb(verb_phrase, includes_subject=True)
    verb_form_with_pobject = prepare_question_verb(verb_phrase, includes_subject=False)

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

        if wh_word_pobject != 'When':
            question.append(preposition.text)

        question = ' '.join(question)

        question += '?'
        questions.append((question, pobject_phrase))

    question = [wh_word_subject]
    question.extend(verb_form_with_pobject)
    question.append(format_phrase(pobject_phrase))

    # question = ' '.join(question)
    question = safe_join(question)
    question += '?'
    questions.append((question, subject_phrase))

    return questions


def generate_agent_questions(match):
    """Only matches the by agent preposition"""
    questions = []
    sentence = match.sentence
    subject = match.get_first_token()
    pobject = match.get_last_token()
    by_prep = pobject.head
    verb = by_prep.head

    subject_phrase = extract_noun_phrase(subject, sentence)
    pobject_phrase = extract_noun_phrase(pobject, sentence)
    verb_phrase = get_verb_phrase(verb, sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_pobject = choose_wh_word(pobject_phrase)

    verb_form_with_subject = prepare_question_verb(verb_phrase, includes_subject=True) + [by_prep.text]
    verb_form_with_pobject = prepare_question_verb(verb_phrase, includes_subject=False) + [by_prep.text]

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
        questions.append((question, pobject_phrase))

    question = [wh_word_subject]
    question.extend(verb_form_with_pobject)
    question.append(format_phrase(pobject_phrase))

    # question = ' '.join(question)
    question = safe_join(question)
    question += '?'
    questions.append((question, subject_phrase))

    return questions


def generate_attribute_questions(match):
    """Gets a match object for the attribute pattern. Returns a list of questions"""
    questions = []
    sentence = match.sentence
    subject = match.get_first_token()
    attribute = match.get_last_token()
    verb = subject.head

    if attribute.head != verb:
        return []

    subject_phrase = extract_noun_phrase(subject, sentence)
    attribute_phrase = extract_noun_phrase(match.get_last_token(), sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_attribute = choose_wh_word(attribute_phrase)
    verb_phrase = get_verb_phrase(verb, sentence)

    verb_form_with_subject = prepare_question_verb(verb_phrase, includes_subject=True)
    verb_form_with_attribute = prepare_question_verb(verb_phrase, includes_subject=False)

    if is_valid_subject(subject_phrase):
        questions.append(('Describe or define ' + format_phrase(subject_phrase) + '.', attribute_phrase))
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
        questions.append((question, attribute_phrase))

    question = [wh_word_subject]
    question.extend(verb_form_with_attribute)
    question.append(format_phrase(attribute_phrase))

    # question = ' '.join(question)
    question = safe_join(question)
    question += '?'
    questions.append((question, subject_phrase))

    return questions


def generate_dobj_questions(match):
    questions = []
    sentence = match.sentence
    direct_object = match.get_last_token()
    subject = match.get_first_token()
    verb = subject.head

    # Means we've matched in error
    if direct_object.head != verb:
        return

    subject_phrase = extract_noun_phrase(subject, sentence)
    direct_object_phrase = extract_noun_phrase(direct_object, sentence)
    verb_phrase = get_verb_phrase(verb, sentence)

    wh_word_subject = choose_wh_word(subject_phrase)
    wh_word_object = choose_wh_word(direct_object_phrase)

    verb_form_with_subject = prepare_question_verb(verb_phrase, includes_subject=True)
    verb_form_with_object = prepare_question_verb(verb_phrase, includes_subject=False)

    if is_valid_subject(subject_phrase):
        questions.append(('Describe the relation or interaction between '
                          + format_phrase(subject_phrase)
                          + ' and '
                          + format_phrase(direct_object_phrase)
                          + '.', verb_phrase))
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
        questions.append((question, direct_object_phrase))

    question = [wh_word_subject]

    question.extend(verb_form_with_object)
    question.append(format_phrase(direct_object_phrase))

    # question = ' '.join(question)
    question = safe_join(question)
    question += '?'
    questions.append((question, subject_phrase))

    return questions


#
# def generate_questions(text):
#
#     # The protocol for getting sentences with corresponding scores and keywords
#     text_as_doc = preprocess.clean_to_doc(text)
#     keywords_with_scores = get_keywords_with_scores(text_as_doc)
#     sentences = preprocess.sentence_tokenize(text)
#     sentences_with_keywords_and_scores = get_sentences_with_keywords_and_scores(sentences, keywords_with_scores)
#
#     sorted_sentences = list(sort_scores(sentences_with_keywords_and_scores))
#     i = 0
#     for sentence in sorted_sentences:
#         # for token in sentence:
#         # print(token.text, token.dep_)
#         # showTree(sentence)
#         print(sentence[0:2], i)
#         i += 1
#         # print(sentences_with_keywords_and_scores[sentence][1])
#         # break
#

def get_coreference(pronoun):
    """Implement if time -> pronoun get document coreference with neuralcoref"""
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
    text_4 = NLP(
        u'A router is a networking device that forwards data packets between computer networks and a house is a living space.')
    d = NLP(
        u'A user centred design process provides a professional approach to creating software with functionality that users need.')

    text = NLP(u"Apple's logo was designed by Steve Jobs in early december 2006 in front of the Empire State Building.")
    text = NLP(u'John never believed that Hamilton shot Aaron Burr.')
    text = NLP(u"He hasn't yet seen the sun.")
    # TODO: THIS CASE NEEDS TO BE TREATED sth ... verb past tense -> is not treated

    text = NLP(u'Morphology: the structure of words')
    text = NLP(u'John thought he would win the prize.')
    text = NLP(u"However, they think this won't work.")
    text = NLP(u"In principle, you've got it all wrong.")
    # What did he come up with?
    text = NLP(u"He came up with the idea of a time machine.")
    text = NLP(u"They were looking forward to finishing the lesson on RISC architecture.")

    text = NLP(u"the essential elements are an")
    text = NLP(u"A car is red. It was seen down the highway.")
    text = NLP(
        u"As Drummond turned his boat seawards and proceeded back to the offshore squadron which was still engaged in an artillery duel with the German defenders, one of the missing launches, ML276 passed her, having caught up with the lost cruiser at this late stage.")
    text = NLP(u"She bought me these books. She got these books for me. I read them one by one.")
    text = NLP(
        u"NLP is essentially multidisciplinary: it is closely related to linguistics (although the extent to which NLP overtly draws on linguistic theory varies considerably)")
    text = NLP(
        u"Like NLP, formal linguistics deals with the development of models of human languages, but some approaches in linguistics reject the validity of statistical techniques, which are seen as an essential part of computational linguistics.")
    text = NLP(u"Compositional semantics is the construction of meaning (often expressed as logic) based on syntax.")
    text = NLP(u"The book, which is now at the store, has sold over 1000 copies so far.")
    text = NLP(u"The computer whose hard disk was broken was taken away.")
    # text = NLP(u"Janoff presented Jobs with several different monochromatic themes for the “bitten” logo, and Jobs immediately took a liking to it.")

    text = NLP(u"Mark's fingerprints were found after the investigation.")
    text = NLP(u'The Bill fo Rights gave the new federal government greater legitimacy.')
    text = NLP(
        u'In the end, it is concluded that the airspeed velocity of a (European) unladen swallow is about 24 miles per hour or 11 meters per second.')
    text = NLP(u'This text is one of the most famous ones in history.')
    text = NLP(u'Darwin studied how species evolve.')
    text = NLP(u'How does this happen?.')
    # text = NLP(u'It is concluded that that the airspeed velocity of a (European) unladen swallow is about 24 miles per hour or 11 meters per second.')

    show_dependencies(text, port=5002)
    # for nc in text.noun_chunks:
    #     print(nc)

    # print(get_phrase(text[7]))


def generate_dobj_pobj_question(match):
    subject = match.get_first_token()
    root_verb = subject.head

    preposition = match.get_token_by_attributes(dependency='prep', head=root_verb)
    preposition_2 = match.get_token_by_attributes(dependency='dative', head=root_verb)
    pobject = match.get_token_by_attributes(dependency='pobj', head=preposition)
    dobject = match.get_token_by_attributes(dependency='dobj', head=root_verb)

    if pobject is None \
            or dobject is None:
        return []

    if preposition is None:
        if preposition_2 is None:
            return []
        else:
            preposition = preposition_2

    subject_phrase = extract_noun_phrase(subject, match.sentence)
    wh_word_subject = choose_wh_word(subject_phrase)

    pobject_phrase = extract_noun_phrase(pobject, match.sentence)
    wh_word_pobject = choose_wh_word(pobject_phrase)

    dobject_phrase = extract_noun_phrase(dobject, match.sentence)
    wh_word_dobject = choose_wh_word(dobject_phrase)

    verb_phrase = get_verb_phrase(root_verb, match.sentence)
    verb_form_with_subject = prepare_question_verb(verb_phrase, includes_subject=True)
    verb_form_with_object = prepare_question_verb(verb_phrase, includes_subject=False)

    questions = []
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

        question.append(format_phrase(dobject_phrase))
        question.append(preposition.text)
        question = ' '.join(question)
        question += '?'
        questions.append((question, pobject_phrase))

        question = [wh_word_dobject]
        if len(verb_form_with_subject) == 1:
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
        else:
            # I don't like this ... could have more than 2 in verb phrase, not in attribute, but other ones yes
            question.append(verb_form_with_subject[0])
            question.append(format_phrase(subject_phrase))
            question.extend(verb_form_with_subject[1:])

        question.append(preposition.text)
        question.append(format_phrase(pobject_phrase))
        # question = ' '.join(question)
        question = safe_join(question)
        question += '?'
        questions.append((question, dobject_phrase))

    question = [wh_word_subject]
    question.extend(verb_form_with_object)
    if pobject.i > dobject.i:
        question.append(format_phrase(dobject_phrase))
        question.append(preposition.text)
        question.append(format_phrase(pobject_phrase))
    else:
        question.append(preposition.text)
        question.append(format_phrase(pobject_phrase))
        question.append(format_phrase(dobject_phrase))

    # question = ' '.join(question)
    question = safe_join(question)
    question += '?'
    questions.append((question, subject_phrase))

    return questions


def generate_q():
    initialize_question_patterns()
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
    # text = NLP(u'The ecclesiastical parish of Navenby was originally placed in the Longoboby Rural Deanery.')
    # text = NLP(u'A router helps with packet forwarding.')
    # text = NLP(u'The handle should be attached before the mantle.')
    # text = NLP(u'The Bill fo Rights gave the new federal government greater legitimacy.')
    # text = NLP(u"The general took his soldiers to the hiding place.")
    text = NLP(u"Apple’s first logo, designed by Jobs and Wayne, depicts Sir Isaac Newton sitting under an apple tree.")
    text = NLP(u"John gave a present to Joanna.")
    text = NLP(u'Darwin studied how species evolve.')
    text = NLP(
        u'During the Gold Rush years in northern California, Los Angeles became known as the "Queen of the Cow Counties" for its role in supplying beef and other foodstuffs to hungry miners in the north.')
    text = NLP(
        u'In the end, it is concluded that the airspeed velocity of a (European) unladen swallow is about 24 miles per hour or 11 meters per second.')
    # text = NLP(u'Almost immediately, though, this was replaced by Rob Janoff’s “rainbow Apple”, the now-familiar rainbow-colored silhouette of an apple with a bite taken out of it.')

    # text = NLP(u'In the end, it is concluded that the airspeed velocity of a (European) unladen swallow is about 24 miles per hour or 11 meters per second.')
    # text = NLP(u"The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.")
    # text = NLP(u"An Apple fell on Newton's head.")
    # text = NLP(u"Mario Kart is the most annoying game to be played.")
    # text = NLP(u"Software that is usable for its purpose is sometimes described by programmers as “intuitive” (easy to learn, easy to remember, easy to apply to new problems) or “powerful” (efficient, effective)")
    # text = NLP(u"This course attempts, so far as possible within 8 lectures, to discuss the important aspects of fields including: Interaction Design, User Experience Design (UX), Interactive Systems Design, Information Visualisation, Cognitive Ergonomics, Man-Machine Interface (MMI), User Interface Design (UI), Human Factors, Cognitive Task Design, Information Architecture (IA), Software Product Design, Usability Engineering, User-Centred Design (UCD) and Computer Supported Collaborative Work (CSCW).")
    # text = NLP(u"Almost immediately, though, this was replaced by Rob Janoff’s “rainbow Apple”, the now-familiar rainbow-colored silhouette of an apple with a bite taken out of it.")

    # show_dependencies(text)
    sentence_object_mock = Sentence(text, 1)
    kw1 = Keyword(sentence_object_mock.as_doc[9], score=0.2, sentence=sentence_object_mock.as_doc)
    kw2 = Keyword(sentence_object_mock.as_doc[10], score=0.1, sentence=sentence_object_mock.as_doc)
    sentence_object_mock.add(kw1)
    sentence_object_mock.add(kw1)
    sentence_object_mock.add(
        KeyPhrase(start_index=9, end_index=10, sentence=sentence_object_mock.as_doc, keywords=[kw1, kw2]))
    generate_questions_trial(trial_sentence=sentence_object_mock, simplify=True, debug=True)


def generate_questions(sentence=None, text=None, trim=True, simplify=False):
    # TODO: normal per sentence, option of including rest of pipeline or not for evaluation
    if sentence and text:
        print("Please only choose one form of input")
        return

    if sentence:
        if isinstance(sentence, Sentence):
            sentences = [sentence]
        elif isinstance(sentence, str):
            sent_as_doc = NLP(sentence)
            sentences = [SentenceProvider(sent_as_doc)]
        elif isinstance(sentence, doc):
            sentences = [SentenceProvider(sentence)]
        else:
            print("Error parsing the input")
            return
    elif text:
        document = preprocess.clean_to_doc(text)
        sentence_provider = SentenceProvider(document)
        sentences = sentence_provider.get_top_sentences(trim=trim)

    else:
        print("No input provided")
        return

    # Start the Matcher
    initialize_question_patterns()
    all_questions = set([])
    seen_questions = set([])
    sentences_with_questions = {}

    for sentence in sentences:
        if simplify:
            sentences = simplifier.simplify_sentence(sentence.text)


def generate_questions_trial(trial_sentence=None, text=None, simplify=False, debug=False):
    if trial_sentence:
        top_sentences = [trial_sentence]

    else:
        if text is None:
            text = read_file(input('Filepath: '))

        document = preprocess.clean_to_doc(text)

        sentence_provider = SentenceProvider(document)

        # resolved_text = resolve_coreferences(document.text)
        # resolved_sentences = [s for s in preprocess.sentence_tokenize(resolved_text[0])]

        # replace_coreferences(resolved_sentences, sentence_provider.sentence_objects)

        top_sentences = sentence_provider.get_top_sentences(trim=False)
        # top_sentences = sentence_provider.get_top_sentences(trim=True)

        # for sent in top_sentences:
        #     print(sent.as_doc)

        trial_sents = [top_sentences[0]]
        # text = clean_and_format(text)
        # coreferences, resolved = get_coreferences(text)

        # text_as_doc = preprocess.clean_and_tokenize(text)

    initialize_question_patterns()
    all_questions = set([])
    seen_questions = set([])

    for sentence in top_sentences:
        if simplify:
            sentences = simplifier.simplify_sentence(sentence.text)
        else:
            sentences = [sentence.as_doc]

        # print("\nOriginal sentence:\n")
        # print(sentence.text)

        # print("\nSimplified sentences:\n")

        for s in sentences:

            matches = MATCHER(s)

            for ent_id, start, end in matches:
                match = Match(ent_id, start, end, s)
                pattern_name = NLP.vocab.strings[ent_id]

                questions = handle_match(pattern_name)(match)
                if questions:
                    for (question, answer) in questions:
                        # q = Question(question, sentence, )
                        if question.lower() not in seen_questions:
                            seen_questions.add(question.lower())
                            q_object = Question(question, sentence, answer)
                            all_questions.add(q_object)

    # print("\nQuestions: \n")

    sorted_questions = sort_by_score(all_questions, descending=True)

    if debug:
        for question in sorted_questions:
            # perplexity = spacy_perplexity(doc=question.content)
            print("Q: {}, score: {}".format(question.content, question.score))
            # print("Perplexity: {}\n".format(perplexity))

    # for question in sorted_questions:
    #     for q2 in sorted_questions:
    #         if question is not q2:
    #             print("Q1: {}\nQ2: {}\nSimilarity:{}".format(question.content, q2.content, question.similarity(q2)))

    return sorted_questions


# Switch statement, sort of
def generare_ccomp_question(match):
    return []


pattern_to_question = {
    "ATTR": lambda match: generate_attribute_questions(match),
    "POBJ": lambda match: generate_prepositional_questions(match),
    "DOBJ": lambda match: generate_dobj_questions(match),
    "AGENT": lambda match: generate_agent_questions(match),
    "DOBJ-POBJ": lambda match: generate_dobj_pobj_question(match),
    "CCOMP": lambda match: generare_ccomp_question(match)
    # TODO: add more patterns
}


def handle_match(pattern_name):
    return pattern_to_question[pattern_name]


def sort_by_score(unsorted, descending=False):
    return sorted(unsorted, key=lambda el: el.score, reverse=descending)


def resolve_coreferences(text):
    coref = Coref(nlp=NLP)

    coref.one_shot_coref(utterances=text)

    resolved_utterance_text = coref.get_resolved_utterances()

    # print(resolved_utterance_text)

    return resolved_utterance_text


def replace_coreferences(resolved_sentences, original_sentence_objects):
    for (i, sentence) in enumerate(resolved_sentences):
        if sentence.text != original_sentence_objects[i].text:
            original_sentence_objects[i].text = sentence.text


if __name__ == '__main__':
    generate_q()
    # trial_sentences()
    # generate_questions_trial(simplify=False)
