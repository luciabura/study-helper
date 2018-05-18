import math
from collections import OrderedDict

from spacy.matcher import Matcher
from spacy.tokens import doc
from wordfreq import word_frequency

import question_generation.sentence_simplifier as simplifier
# from evaluation.question_evaluation import spacy_perplexity
from neuralcoref import Coref
from question_generation import *
from question_generation.question import Question
from sentence_extraction.sentence_provider import SentenceProvider
from sentence_extraction.sentence import Sentence
# from summarization.summary import get_sentences_with_keywords_and_scores
from text_processing import preprocessing as preprocess
from text_processing.grammar import *
from utilities import NLP
from utilities.read_write import read_file

WHO_ENTS = ['PERSON']
WHEN_ENTS = ['DATE']
WHERE_ENTS = ['LOCATION', 'FACILITY', 'ORG', 'LOC', 'GPE']
HOW_MUCH_ENTS = ['MONEY', 'PERCENT']
WHEN_PREPS = ['before', 'after', 'since', 'until', 'when']
WHERE_PREPS = ['to', 'on', 'at', 'over', 'in', 'behind', 'above', 'below', 'from', 'inside', 'outside']

MATCHER = Matcher(NLP.vocab)


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
    direct_object_d = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, {DEP: 'dobj'}]
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
    MATCHER.add("DOBJ", None, direct_object_d)
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


def sort_scores(scores):
    sorted_scores = OrderedDict(sorted(scores.items(), key=lambda t: t[1], reverse=True))

    return sorted_scores

# =========================== VERB PREPARATION ===========================


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

# =========================== QG PATTERN EXTRACTION ===========================


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
    if any(tok.ent_type_ in WHO_ENTS for tok in span):
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


def generare_ccomp_question(match):
    return []


# =========================== HANDLING COREFERENCES ===========================


def get_coreference(pronoun):
    """Implement if time -> pronoun get document coreference with neuralcoref"""
    return 0


def score_and_filter(questions, descending=True):
    """
    Expects a set of question objects, all assigned to the same sentence
    Will return a descending list of questions, preferably not similar
    :param questions:
    :param descending:
    :return:
    """
    similarity_threshold = 0.92
    sorted_questions = sort_by_score(questions, descending=descending)
    questions = set([])
    while sorted_questions:
        current = sorted_questions.pop()
        questions.add(current)

        new_sorted_questions = set([])

        for q in sorted_questions:
            if current.content.similarity(q.content) < similarity_threshold:
                new_sorted_questions.add(q)

        sorted_questions = new_sorted_questions

    return questions


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
            sentences = simplifier.simplify_sentence_1(sentence.text)
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


# =========================== PATTERN HANDLING ===========================

pattern_to_question = {
    "ATTR": lambda match: generate_attribute_questions(match),
    "POBJ": lambda match: generate_prepositional_questions(match),
    "DOBJ": lambda match: generate_dobj_questions(match),
    "AGENT": lambda match: generate_agent_questions(match),
    "DOBJ-POBJ": lambda match: generate_dobj_pobj_question(match),
    "CCOMP": lambda match: generare_ccomp_question(match)
}


def handle_match(pattern_name):
    return pattern_to_question[pattern_name]

# =========================== PATTERN HANDLING ===========================


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


# =========================== MAIN QUESTION GENERATION ==========================


def print_sentences_with_questions(sentences_with_questions, with_keys=False):
    for sent in sentences_with_questions.keys():
        if sentences_with_questions[sent]:
            print("\nSentence: {}\n".format(sent.text))
            if with_keys:
                print("Key-phrases: {}\n".format([kp.text for kp in sent.key_phrases]))

            for question in sentences_with_questions[sent]:
                print("Question: {}".format(question.content))


def get_sentences_with_questions(sentences, simplify=False):
    initialize_question_patterns()

    seen_questions = set([])
    sentences_with_questions = {}

    for sentence in sentences:
        sentences_with_questions[sentence] = set([])
        if simplify:
            simplified_sentences = simplifier.simplify_sentence_2(sentence.text)
        else:
            simplified_sentences = [NLP(sentence.text)]

        for s in simplified_sentences:
            matches = MATCHER(s)

            for ent_id, start, end in matches:
                match = Match(ent_id, start, end, s)
                pattern_name = NLP.vocab.strings[ent_id]

                questions = handle_match(pattern_name)(match)
                if questions:
                    for (question, answer) in questions:
                        if question.lower() not in seen_questions:
                            seen_questions.add(question.lower())
                            q_object = Question(question, sentence, answer)
                            sentences_with_questions[sentence].add(q_object)

    return sentences_with_questions


def generate_questions(sentence=None, text=None, trim=True, simplify=False, verbose=False):
    """
    Generates questions from either a single sentence(str or object) or a full text.
    :param sentence: String or Sentence object
    :param text: String
    :param trim: boolean, if True, use only top scoring ("relevant") sentences from the previous step
    :param simplify: boolean, if True, apply sentence simplification step
    :param verbose: boolean, if True, print the output to console
    :return: A dictionary of questions in sorted order of their importance in the text
    """
    if sentence and text:
        print("Please only choose one form of input")
        return

    if sentence:
        if isinstance(sentence, Sentence):
            sentences = [sentence]
        elif isinstance(sentence, str):
            sent_as_doc = NLP(sentence)
            sentences = SentenceProvider(sent_as_doc).sentence_objects
        elif isinstance(sentence, doc):
            sentences = SentenceProvider(sentence).sentence_objects
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
    sentences_with_questions = get_sentences_with_questions(sentences, simplify=simplify)
    all_questions = set([])

    for sent in sentences_with_questions.keys():
        questions = score_and_filter(sentences_with_questions[sent], descending=True)
        all_questions.union(questions)

    if verbose:
        print_sentences_with_questions(sentences_with_questions)

    all_sorted_questions = sort_by_score(all_questions, descending=True)

    return all_sorted_questions


class QuestionProvider:
    def __init__(self, text, topic=None):
        self.text = text
        self.sentence_provider = SentenceProvider(preprocess.clean_to_doc(text), topic=topic)
        self.sentences_with_questions = None
        self.ordered_sentences = self.sentence_provider.get_top_sentences(trim=True)

        self.key_phrases = self.sentence_provider.key_phrases
        self.questions = None
        self.sents_only = False
        self.keys_only = False

    def print_questions(self, verbose=False):
        if self.keys_only:
            print("Key-phrases extracted from the text:")
            print([kp.text for kp in self.key_phrases])
        if self.sents_only:
            print("Sentences extracted from the text, in order of importance:")
            for sent in self.ordered_sentences:
                print(sent.text)
        if not (self.keys_only or self.sents_only):
            if verbose:
                print("Showing the sentences and their corresponding questions:")
                self.sentences_with_questions = get_sentences_with_questions(self.sentence_provider.sentence_objects, simplify=False)
                print_sentences_with_questions(self.sentences_with_questions, with_keys=True)
            else:
                self.questions = list(generate_questions(self.text))
                for question in self.questions:
                    print(question.text)


# if __name__ == '__main__':
#     # generate_q()
#     # trial_sentences()
#     # generate_questions_trial(simplify=False)
#     text = read_file(input('Filepath: '))
#     generate_questions(text=text, trim=False, verbose=True, simplify=True)
