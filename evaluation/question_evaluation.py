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
from evaluation.summary_evaluation import NLTK_BLEU
from question_generation.question_provider import generate_questions_trial
import evaluation.heilman as heilman

from text_processing.grammar import spacy_similarity
from utilities.read_write import print_to_file

NLP = spacy.load('en_core_web_lg')

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
questions_filepath = os.path.join(__location__, 'questions_corpus.txt')
squad_filepath = os.path.join(__location__, 'paragraphs_with_questions.json')
msmarco_filepath = os.path.join(__location__, 'sentences_with_questions.json')
results_filepath = os.path.join(__location__, 'results')

BLEU = BLEUCalculator()

with open(squad_filepath) as f:
    squad = json.load(f)

with open(msmarco_filepath) as f:
    msmarco = json.load(f)


def calculate_similarity(q1, q2):
    semantic_similarity = spacy_similarity(q1, q2)
    sintactic_similarity = NLTK_BLEU.sentence_bleu([q1], q2)

    overall_similarity = (2.0 * semantic_similarity * sintactic_similarity) \
                         / (semantic_similarity + sintactic_similarity)

    return overall_similarity


def evaluate_question_similarity_squad():
    # orginal_text, generated_questions, reference_questions
    """Evaluate reference questions against system generated questions in terms
    of semantic similarity -- using spacy similarity
    How: max semantic similarity if they have the same answer phrase"""

    data = squad['data']
    i = 0
    for data_el in data:
        title = data_el['title']
        text = data_el['text']
        golden_questions = []
        my_questions = [q.content.text for q in generate_questions_trial(text=text, simplify=False)]
        heilman_questions = [question for question in heilman.generate_questions_from_path(text=text)]

        for question in data_el['questions']:
            golden_questions.append(question)

        av_me = 0
        av_he = 0
        for q1 in golden_questions:
            max_sim = 0
            for q2 in my_questions:
                sim = calculate_similarity(q1, q2)
                if sim > max_sim:
                    max_sim = sim

            av_me += max_sim

            max_sim = 0
            for q2 in heilman_questions:
                sim = calculate_similarity(q1, q2)
                if sim > max_sim:
                    max_sim = sim

            av_he += max_sim

        av_me = av_me / len(golden_questions)
        av_he = av_he / len(golden_questions)

        print("My system:{},Heilman:{}".format(av_me, av_he).replace(",", "\n"))

        i += 1
        if i == 4:
            break


def evaluate_question_similarity_msmarco():
    # orginal_text, generated_questions, reference_questions
    """Evaluate reference questions against system generated questions in terms
    of semantic similarity -- using spacy similarity
    How: max semantic similarity if they have the same answer phrase"""

    data = msmarco['data']
    sys_score = 0
    heil_score = 0
    count = 0
    for data_el in data:
        count += 1
        question = data_el['question']
        text = data_el['answer']
        golden_questions = [question]
        my_questions = [q.content.text for q in generate_questions_trial(text=text, simplify=False)]
        string_qs = '\n'.join(my_questions)
        file_path = os.path.join(results_filepath, "system_"+str(count))
        print_to_file(path=file_path, content=string_qs)
        heilman_questions = [question for question in heilman.generate_questions_from_path(text=text)]

        string_hs = '\n'.join(heilman_questions)
        file_path = os.path.join(results_filepath, "heilman_"+str(count))
        print_to_file(path=file_path, content=string_hs)

        av_me = 0
        av_he = 0
        for q1 in golden_questions:
            max_sim = 0
            for q2 in my_questions:
                sim = calculate_similarity(q1, q2)
                if sim > max_sim:
                    max_sim = sim

            av_me += max_sim

            max_sim = 0
            for q2 in heilman_questions:
                sim = calculate_similarity(q1, q2)
                if sim > max_sim:
                    max_sim = sim

            av_he += max_sim

        av_me = av_me / len(golden_questions)
        sys_score += av_me

        av_he = av_he / len(golden_questions)
        heil_score += av_he

        print("My system:{},Heilman:{}".format(av_me, av_he).replace(",", "\n"))

    sys_score /= count
    heil_score /= count
    print("My system:{},Heilman:{}".format(sys_score, heil_score).replace(",", "\n"))


def spacy_perplexity(text=None, doc=None, model=NLP):
    if doc is None:
        if text is None:
            return
        doc = model(text)

    log_sum = 0
    for token in doc:
        log_sum -= token.prob
        # print(token.text, token.prob)

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
    evaluate_question_similarity_msmarco()
    # evaluate_question_similarity_squad()
