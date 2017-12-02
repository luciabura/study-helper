"""
Unit tests for the baseline for keyword extraction
"""
from baselines.keyword_baseline import get_keywords, get_most_common


def test_get_most_common():
    """
    Unit text to check correct count of most common
    """
    words = ['a', 'b', 'b', 'b', 'd', 'd', 'e', 'c', 'c', 'c']
    expected_most_common_2 = ['b', 'c']

    most_common_2 = get_most_common(words, 2)
    assert sorted(expected_most_common_2) == sorted(most_common_2)


def test_get_keywords():
    """
    Unit test to check correct return
    """
    text = "Computer science is the study of the theory, " \
           "experimentation, and engineering that form the basis for " \
           "the design and use of a computer."

    expected_keyword = ["computer"]

    result_keyword = get_keywords(text, 1)

    assert result_keyword == expected_keyword
