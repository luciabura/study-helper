import pytest

from question_generation.question_provider import score_and_filter, prepare_question_verb, get_verb_form
from question_generation.question import Question, sequence_surprize
from sentence_extraction.sentence import Sentence
from utilities import NLP

@pytest.mark.parametrize(
    "sentence,verb_indexes,expected_verb_X,expected_verb_Y", [
        (u'X is Y.', (1,2), ['is'], ['is']),
        (u'X saw Y.', (1, 2), ['did', 'see'], ['saw']),
        (u'X has seen Y.', (1, 3), ['has', 'seen'], ['has', 'seen']),
        (u'X have seen Y.', (1, 3), ['have', 'seen'], ['has', 'seen']),
        (u'X never saw Y.', (1, 3), ['did', 'never', 'see'], ['never', 'saw']),
        (u"X didn't see Y.", (1, 4), ['did', 'not', 'see'], ['did', 'not', 'see']),
        (u'X never did see Y.', (1, 4), ['did', 'never', 'see'], ['never', 'did', 'see']),
        (u'X did never see Y.', (1, 4), ['did', 'never', 'see'], ['did', 'never', 'see']),
        (u'X had never seen Y.', (1, 4), ['had', 'never', 'seen'], ['had', 'never', 'seen']),
        (u'X had really liked Y.', (1, 4), ['had', 'really', 'liked'], ['had', 'really', 'liked']),
        (u"They've seen Y", (1, 3), ['have', 'seen'], ['has', 'seen']),
    ]
)
def test_prepare_question_verb(sentence, verb_indexes, expected_verb_X, expected_verb_Y):
    sent = NLP(sentence)
    vp_original = sent[verb_indexes[0]:verb_indexes[1]]

    verb_for_X = prepare_question_verb(vp_original, includes_subject=True)
    verb_for_Y = prepare_question_verb(vp_original, includes_subject=False)

    assert verb_for_X == expected_verb_X
    assert verb_for_Y == expected_verb_Y


@pytest.mark.parametrize(
    "verb,expected_verb_with_helper,expected_verb_no_helper", [
        (u'was', ['was'], ['was']),
        (u'is', ['is'], ['is']),
        (u'saw', ['did', 'see'], ['saw']),
        (u'went', ['did', 'go'], ['went']),
    ]
)
def test_get_correct_verb_helper(verb, expected_verb_with_helper, expected_verb_no_helper):
    verb_original = NLP(verb)[0]

    verb_with_helper = get_verb_form(verb_original, includes_subject=True)
    verb_no_helper = get_verb_form(verb_original, includes_subject=False)

    assert verb_with_helper == expected_verb_with_helper
    assert verb_no_helper == expected_verb_no_helper


def test_get_correct_verb_helper_none():
    verb_original = None

    verb_with_helper = get_verb_form(verb_original, includes_subject=True)
    verb_no_helper = get_verb_form(verb_original, includes_subject=False)

    assert verb_with_helper is None
    assert verb_no_helper is None


def test_sort_and_filter():
    sent = Sentence(NLP("John is at the supermarket."), 1)

    q1 = Question("Where is John?", sent, None)
    q1.score = 2.0

    q2 = Question("Who is at the supermaket?", sent, None)
    q2.score = 1.0

    q3 = Question("Where be John?", sent, None)
    q3.score = 1.9

    qs = {q1, q2, q3}

    filtered_questions = score_and_filter(qs)

    assert filtered_questions == {q1, q2}
