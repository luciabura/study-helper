import glob
import os

from sumeval.metrics.bleu import BLEUCalculator
from sumeval.metrics.rouge import RougeCalculator

from evaluation.py3cocoeval.bleu.bleu_scorer import BleuScorer
from evaluation.py3cocoeval.meteor.meteor import Meteor

from data import DATA_DIR
from summarization import gensim_sum, sentence_provider, sumy_summary
from text_processing import preprocessing
from utilities import NLP
from utilities.read_write import read_file, print_to_file, print_summary_to_file
from baselines.summary_baseline import get_summary as baseline_summary

import nltk.translate.bleu_score as NLTK_BLEU

SUMM_DIR = os.path.join(DATA_DIR, 'summarization_eval')
SUMM_BODY = os.path.join(SUMM_DIR, 'body')
SUMM_GOLD = os.path.join(SUMM_DIR, 'summary_gold')
SUMM_SYSTEM = os.path.join(SUMM_DIR, 'system')
SUMM_SCORES = os.path.join(SUMM_DIR, 'scores')

OP_SUMM_DIR = os.path.join(DATA_DIR, 'summarization_eval/opinionis')
OP_SUMM_BODY = os.path.join(OP_SUMM_DIR, 'body')
OP_SUMM_GOLD = os.path.join(OP_SUMM_DIR, 'summary_gold')
OP_SUMM_SYSTEM = os.path.join(OP_SUMM_DIR, 'system')
OP_SUMM_SCORES = os.path.join(OP_SUMM_DIR, 'scores')

ROUGE = RougeCalculator(stopwords=True, lang="en")
BLEU = BLEUCalculator()


def make_summaries(summarizer, identifier):
    os.chdir(SUMM_BODY)
    for file in glob.glob("*body.txt"):
        input_path = os.path.join(SUMM_BODY, file)
        file_text = read_file(input_path)
        summ = summarizer(file_text)
        print_summary_to_file(summ, input_path, SUMM_SYSTEM, identifier=identifier)


def make_summaries_OP(summarizer, identifier):
    os.chdir(OP_SUMM_BODY)
    for file in glob.glob("*.txt.data"):
        input_path = os.path.join(OP_SUMM_BODY, file)
        file_text = read_file(input_path, encoding="ISO-8859-1")
        summ = summarizer(file_text)
        print_summary_to_file(summ, input_path, OP_SUMM_SYSTEM, identifier=identifier)


def evaluate_summaries_OP(summarizer_identifier):
    os.chdir(OP_SUMM_SYSTEM)
    n = 0
    av_r1 = av_r2 = av_rl = av_rbe = av_b_1 = av_b_2 = av_b = 0
    for file in glob.glob("*summary" + summarizer_identifier + ".txt"):
        n = n + 1
        summary_input_path = os.path.join(SUMM_SYSTEM, file)
        reference_input_path = os.path.join(SUMM_GOLD, file.replace(summarizer_identifier, ''))

        system_summary = read_file(summary_input_path)
        reference_summary = read_file(reference_input_path)

        # result = get_evaluation_scores(system_summary, reference_summary)
        r1, r2, rl, rbe, b_1, b_2, b = get_evaluation_scores(system_summary, reference_summary)
        av_r1 += r1
        av_r2 += r2
        av_rl += rl
        av_rbe += rbe
        av_b_1 += b_1
        av_b_2 += b_2
        av_b += b

        # output_path = os.path.join(SUMM_SCORES, file.replace('summary', '_scores'))
        # print_to_file(result, output_path)

    av_r1 /= n
    av_r2 /= n
    av_rl /= n
    av_rbe /= n
    av_b_1 /= n
    av_b_2 /= n
    av_b /= n

    result = ("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}, "
              "BLEU-1: {}, BLEU-2: {}, BLEU-nltk : {}".format(
                     av_r1, av_r2, av_rl, av_rbe, av_b_1, av_b_2, av_b).replace(", ", "\n"))
    name = "evaluation_summary" + summarizer_identifier + ".txt"
    output_path = os.path.join(SUMM_SCORES, name)
    print_to_file(result, output_path)


def evaluate_summaries(summarizer_identifier):
    os.chdir(SUMM_SYSTEM)
    n = 0
    av_r1 = av_r2 = av_rl = av_rbe = av_b_1 = av_b_2 = 0
    summaries = []
    references = []
    for file in glob.glob("*summary" + summarizer_identifier + ".txt"):
        n = n + 1
        summary_input_path = os.path.join(SUMM_SYSTEM, file)
        reference_input_path = os.path.join(SUMM_GOLD, file.replace(summarizer_identifier, ''))

        system_summary = read_file(summary_input_path)
        summaries.append(system_summary)
        reference_summary = read_file(reference_input_path)
        references.append([reference_summary])

        result = get_evaluation_scores(system_summary, reference_summary)
        r1, r2, rl, rbe, b_1, b_2 = get_evaluation_scores(system_summary, reference_summary)
        av_r1 += r1
        av_r2 += r2
        av_rl += rl
        av_rbe += rbe
        av_b_1 += b_1
        av_b_2 += b_2

        # output_path = os.path.join(SUMM_SCORES, file.replace('summary', '_scores'))
        # print_to_file(result, output_path)

    av_r1 /= n
    av_r2 /= n
    av_rl /= n
    av_rbe /= n
    av_b_1 /= n
    av_b_2 /= n

    bleu_score = NLTK_BLEU.corpus_bleu(references, summaries, smoothing_function=NLTK_BLEU.SmoothingFunction().method2)

    result = ("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}, "
              "BLEU-1: {}, BLEU-2: {}, BLEU-nltk : {}".format(
                     av_r1, av_r2, av_rl, av_rbe, av_b_1, av_b_2, bleu_score).replace(", ", "\n"))
    name = "evaluation_summary" + summarizer_identifier + ".txt"
    output_path = os.path.join(SUMM_SCORES, name)
    print_to_file(result, output_path)


def prepare_for_eval(summary):
    sentences = summary.split('\n')
    return sentences


def get_evaluation_scores(summary, reference):
    rouge_1 = ROUGE.rouge_n(
                summary=summary,
                references=reference,
                n=1)

    rouge_2 = ROUGE.rouge_n(
                summary=summary,
                references=reference,
                n=2)

    rouge_l = ROUGE.rouge_l(
                summary=summary,
                references=reference)

    rouge_be = ROUGE.rouge_be(
                summary=summary,
                references=reference)

    # bleu = BLEU.bleu(summary=summary,
    #                  references=reference)

    # result = ("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}, BLEU: {}".format(
    #             rouge_1, rouge_2, rouge_l, rouge_be, bleu).replace(", ", "\n"))

    # return result
    [bleu_1, bleu_2, _, _] = BleuScorer(test=summary, refs=[reference]).compute_score()

    return rouge_1, rouge_2, rouge_l, rouge_be, bleu_1, bleu_2


def evaluate_summary_manual():
    summary_path = input("Text to summarize path:")
    summary_text = read_file(summary_path)

    reference_path = input("Reference summary path:")
    reference_summary = read_file(reference_path)
    doc_reference = NLP(reference_summary)
    ref_sentences = [sent.string for sent in doc_reference.sents]
    sentence_count = len(ref_sentences)

    document = preprocessing.clean_to_doc(summary_text)
    summarizer = sentence_provider.SentenceProvider(document)

    system_summary = summarizer.get_summary(sentence_count=sentence_count)
    gensim_summary = gensim_sum.get_summary(summary_text)

    result = get_evaluation_scores(system_summary, reference_summary)
    print("System summary")
    print(result)

    result = get_evaluation_scores(gensim_summary, reference_summary)
    print("State of the art summary")
    print(result)


# def evaluate_summarizer(summarizer):
#     summary_path = input("Text to summarize path:")
#     summary_text = read_file(summary_path)
#
#     system_summary = summarizer(summary_text)
#
#     reference_path = input("Reference summary path:")
#     reference_summary = read_file(reference_path)
#
#     result = get_evaluation_scores(system_summary, reference_summary)
#
#     print(result)


if __name__ == '__main__':
    # make_summaries(summarizer=gensim_sum.get_summary, identifier='_A')
    # make_summaries(summarizer=summary.get_summary, identifier='_B')
    # make_summaries(summarizer=summary.get_summary, identifier='_B2')
    make_summaries(summarizer=sentence_provider.get_summary, identifier='_C')
    # make_summaries_OP(summarizer=sentence_provider.get_summary, identifier='_C')
    # make_summaries(summarizer=baseline_summary, identifier='_baseline')
    # make_summaries(summarizer=sumy_summary.get_summary, identifier='_E')
    # make_summaries(summarizer=sumy_summary.get_summary, identifier='_D')
    # make_summaries(summarizer=sumy_summary.get_summary, identifier='_F')
    # evaluate_summaries(summarizer_identifier='_A')
    # evaluate_summaries(summarizer_identifier='_B')
    # evaluate_summaries(summarizer_identifier='_B2')
    evaluate_summaries(summarizer_identifier='_C')
    # evaluate_summaries(summarizer_identifier='_D')
    # evaluate_summaries(summarizer_identifier='_E')
    # evaluate_summaries(summarizer_identifier='_F')
    # evaluate_summaries(summarizer_identifier='_baseline')
    # evaluate_summary_manual()
