import glob
import os

from sumeval.metrics.bleu import BLEUCalculator
from sumeval.metrics.rouge import RougeCalculator

from data import DATA_DIR
from summarization import gensim_sum, summary
from utilities.utils import read_file, print_file, print_summary_to_file

SUMM_DIR = os.path.join(DATA_DIR, 'summarization_eval')
SUMM_BODY = os.path.join(SUMM_DIR, 'body')
SUMM_GOLD = os.path.join(SUMM_DIR, 'summary_gold')
SUMM_SYSTEM = os.path.join(SUMM_DIR, 'system')
SUMM_SCORES = os.path.join(SUMM_DIR, 'scores')

ROUGE = RougeCalculator(stopwords=True, lang="en")
BLEU = BLEUCalculator()


def make_summaries(summarizer, identifier):
    os.chdir(SUMM_BODY)
    for file in glob.glob("*body.txt"):
        input_path = os.path.join(SUMM_BODY, file)
        print_summary_to_file(summarizer, input_path, SUMM_SYSTEM, identifier=identifier)


def evaluate_summaries(summarizer_identifier):
    os.chdir(SUMM_SYSTEM)
    for file in glob.glob("*summary" + summarizer_identifier + ".txt"):
        summary_input_path = os.path.join(SUMM_SYSTEM, file)
        reference_input_path = os.path.join(SUMM_GOLD, file.replace(summarizer_identifier, ''))

        system_summary = read_file(summary_input_path)
        reference_summary = read_file(reference_input_path)

        result = get_evaluation_scores(system_summary, reference_summary)
        output_path = os.path.join(SUMM_SCORES, file.replace('summary', '_scores'))
        print_file(result, output_path)


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

    bleu = BLEU.bleu(summary=summary,
                     references=reference)

    result = ("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}, BLEU: {}".format(
                rouge_1, rouge_2, rouge_l, rouge_be, bleu).replace(", ", "\n"))

    return result


if __name__ == '__main__':
    make_summaries(summarizer=gensim_sum.get_summary, identifier='_A')
    make_summaries(summarizer=summary.get_summary, identifier='_B')
    evaluate_summaries(summarizer_identifier='_A')
    evaluate_summaries(summarizer_identifier='_B')
