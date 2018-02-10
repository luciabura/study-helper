"""
Takes qq system output
Heilman QG output

Measures:
1) syntactical correctness
2) Similarity with gold questions -> spacy similarity + words present + phrase sequence
3*) relevance to topic if topic exists
"""

import math

import spacy

NLP = spacy.load('en_core_web_lg')


def spacy_perplexity(text, model=NLP):
    doc = model(text)
    log_sum = 0
    for token in doc:
        log_sum -= token.prob

    if len(doc) > 0:
        log_sum /= len(doc)

    perplexity = math.pow(2, log_sum)
    return perplexity


def test_spacy_perplexity():
    bad_grammar = "What are John doing?"
    good_grammar = "What is John doing?"

    assert spacy_perplexity(bad_grammar, NLP) > spacy_perplexity(good_grammar, NLP)


if __name__ == '__main__':
    text = "What is the colour of his hair?"
    text2 = "What are John doing?"
    text3 = "What is John doing?"
    text4 = "Why is John going to the supermarket?"
    text5 = "Where give Mark to Ana?"

    print(spacy_perplexity(text, NLP))
    print(spacy_perplexity(text2, NLP))
    print(spacy_perplexity(text3, NLP))
    print(spacy_perplexity(text4, NLP))
    print(spacy_perplexity(text5, NLP))


def similarity_overlap_score(q1, q2):
    """Takes two questions as text"""
    pass


def semantic_overlap_score(s1, s2):
    pass
