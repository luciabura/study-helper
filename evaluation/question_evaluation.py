"""
Takes qq system output
Heilman QG output

Measures:
1) syntactical correctness
2) Similarity with gold questions -> spacy similarity + words present + phrase sequence
3*) relevance to topic if topic exists
"""
import json
import math

import os
import spacy
from sumeval.metrics.bleu import BLEUCalculator

from question_generation.questions_2 import generate_questions_trial
from text_processing.grammar import spacy_similarity

NLP = spacy.load('en_core_web_lg')

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
questions_filepath = os.path.join(__location__, 'questions_corpus.txt')
json_filepath = os.path.join(__location__, 'paragraphs_with_questions.json')
BLEU = BLEUCalculator()

with open(json_filepath) as f:
    data_file = json.load(f)


def calculate_similarity(q1, q2):
    semantic_similarity = spacy_similarity(q1, q2)
    sintactic_similarity = BLEU.bleu(q1, q2)

    print(sintactic_similarity)

    overall_similarity = (2.0 * semantic_similarity * sintactic_similarity) \
                         / (semantic_similarity + sintactic_similarity)

    return overall_similarity


def evaluate_question_similarity():
    # orginal_text, generated_questions, reference_questions
    """Evaluate reference questions against system generated questions in terms
    of semantic similarity -- using spacy similarity
    How: max semantic similarity if they have the same answer phrase"""

    data = data_file['data']
    i = 0
    for data_el in data:
        title = data_el['title']
        text = data_el['text']
        golden_questions = []
        my_questions = [q.content for q in generate_questions_trial(text=text)]

        for question in data_el['questions']:
            golden_questions.append(question)

        av = 0
        for q1 in golden_questions:
            max_sim = 0
            for q2 in my_questions:
                sim = calculate_similarity(q1, q2)
                if sim > max_sim:
                    max_sim = sim
            av += max_sim

        av = av / len(golden_questions)
        print(av)
        break
        # i += 1
        # if i == 4:
        #     break


def spacy_perplexity(text, model=NLP):
    doc = model(text)
    log_sum = 0
    for token in doc:
        log_sum -= token.prob

    if len(doc) > 0:
        log_sum /= len(doc)

    perplexity = math.pow(2, log_sum)
    return perplexity


# def test_spacy_perplexity():
#     bad_grammar = "What are John doing?"
#     good_grammar = "What is John doing?"
#
#     assert spacy_perplexity(bad_grammar, NLP) > spacy_perplexity(good_grammar, NLP)


if __name__ == '__main__':
    # text = "What is the colour of his hair?"
    # text2 = "What are John doing?"
    # text3 = "What is John doing?"
    # text4 = "Why is John going to the supermarket?"
    # text5 = "Where give Mark to Ana?"
    # orginal_text, generated_questions, reference_questions
    # print(spacy_perplexity(text, NLP))
    # print(spacy_perplexity(text2, NLP))
    # print(spacy_perplexity(text3, NLP))
    # print(spacy_perplexity(text4, NLP))
    # print(spacy_perplexity(text5, NLP))
    evaluate_question_similarity()


def similarity_overlap_score(q1, q2):
    """Takes two questions as text"""
    pass


def semantic_overlap_score(s1, s2):
    pass
