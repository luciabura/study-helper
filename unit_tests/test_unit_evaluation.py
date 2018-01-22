import sys
from evaluation.keywords_evaluation import keyword_score, keyphrase_score

EPSILON = sys.float_info.epsilon


def test_keyword_score_empty():
    keywords = []
    ref_keywords = ['test']

    f_mes, prec, rec = keyword_score(keywords, ref_keywords)

    assert f_mes == 0
    assert prec == 0
    assert rec == 0


def test_keyword_score():
    keywords = ['test', 'test2']
    ref_keywords = ['test']

    f_mes, prec, rec = keyword_score(keywords, ref_keywords)

    expected_prec = 0.5
    expected_rec = 1
    alpha = 0.5
    expected_f1 = 1.0 / (alpha / expected_prec + (1 - alpha) / expected_rec)

    assert abs(prec - expected_prec) < EPSILON
    assert abs(rec - expected_rec) < EPSILON
    assert abs(f_mes - expected_f1) < EPSILON


def test_keyphrase_score_big_test_keyphrase():
    keywords = ['test case']
    ref_keywords = ['test']

    f_mes, prec, rec = keyphrase_score(keywords, ref_keywords)

    expected_prec = 0.5
    expected_rec = 1
    alpha = 0.5
    expected_f1 = 1.0 / (alpha / expected_prec + (1 - alpha) / expected_rec)

    assert abs(prec - expected_prec) < EPSILON
    assert abs(rec - expected_rec) < EPSILON
    assert abs(f_mes - expected_f1) < EPSILON


def test_keyphrase_score_big_reference_keyphrase():
    keywords = ['test']
    ref_keywords = ['test case']

    f_mes, prec, rec = keyphrase_score(keywords, ref_keywords)

    expected_prec = 1
    expected_rec = 0.5
    alpha = 0.5
    expected_f1 = 1.0 / (alpha / expected_prec + (1 - alpha) / expected_rec)

    assert abs(prec - expected_prec) < EPSILON
    assert abs(rec - expected_rec) < EPSILON
    assert abs(f_mes - expected_f1) < EPSILON