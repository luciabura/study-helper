from keyword_extraction import keywords
from keyword_extraction import keywords_TR_lem

from text_processing.preprocessing import NLP


def test_build_lemmas():
    words = ['phone', 'phones', 'magical', 'magic']
    expected_lemmas = ['phone', 'magic', 'magical']
    expected_l2w = {'phone': ['phone', 'phones'],
                    'magic': ['magic'],
                    'magical': ['magical']}
    expected_w2l = {'phone': ['phone'],
                    'phones': ['phone'],
                    'magic': ['magic'],
                    'magical': ['magical']}

    words_to_lemmas, lemmas_to_words = keywords.build_lemmas(words)
    result_lemmas = list(lemmas_to_words.keys())
    result_words = list(words_to_lemmas.keys())

    assert sorted(result_lemmas) == sorted(expected_lemmas)
    assert sorted(result_words) == sorted(words)

    assert sorted(expected_l2w) == sorted(lemmas_to_words)
    assert sorted(expected_w2l) == sorted(words_to_lemmas)


def test_get_graph_tokens():
    text = 'This is a test sentence.'
    # No graph words
    INCLUDE_POS_FILTER = []
    tokens = [token for token in NLP(text)]

    graph_tokens = keywords_TR_lem.get_graph_tokens(tokens, INCLUDE_POS_FILTER)
    expected_graph_tokens = []

    assert graph_tokens == expected_graph_tokens

    # Only nouns
    INCLUDE_POS_FILTER = ['NN']
    graph_tokens = keywords_TR_lem.get_graph_tokens(tokens, INCLUDE_POS_FILTER)
    expected_graph_tokens = [tokens[3], tokens[4]]

    assert graph_tokens == expected_graph_tokens


def test_add_graph_edges():
    # TODO:
    return True


def test_build_graph():
    # TODO:
    return True


