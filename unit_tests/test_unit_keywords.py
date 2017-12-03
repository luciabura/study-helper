from keyword_extraction import keywords


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
    result_lemmas = lemmas_to_words.keys()
    result_words = words_to_lemmas.keys()

    assert sorted(result_lemmas) == sorted(expected_lemmas)
    assert sorted(result_words) == sorted(words)

    assert sorted(expected_l2w) == sorted(lemmas_to_words)
    assert sorted(expected_w2l) == sorted(words_to_lemmas)
