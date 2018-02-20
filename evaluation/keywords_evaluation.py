import glob
import os

from gensim.summarization import keywords as gensim_keywords
from nltk.metrics.scores import f_measure, precision, recall

from baselines.keyword_baseline import get_keywords as get_baseline_keywords
from data import DATA_DIR
from keyword_extraction.keywords_TR import get_keywords as get_graph_keywords_2
from keyword_extraction.keywords_TR_lem import KeywordProvider
from keyword_extraction.keywords_filtered import get_keywords as get_graph_keywords_4
from text_processing import preprocessing
from text_processing.preprocessing import clean_and_tokenize
from utilities.read_write import read_file, print_to_file

KEYWORD_DIR = os.path.join(DATA_DIR, 'keywords_eval')


def keyphrase_f1(keys, ref_keys, alpha=0.5):
    """
    Given a set of reference values and a set of test values, return
    the f-measure of the test values, when compared against the
    reference values.  The f-measure is the harmonic mean of the
    ``precision`` and ``recall``, weighted by ``alpha``.  In particular,
    given the precision *p* and recall *r* defined by:

    The f-measure is:

    - *1/(alpha/p + (1-alpha)/r)*

    :param keys: Set of values to compare against reference values
    :param ref_keys: Reference values
    :param alpha: weight
    :return:
    """

    p = keyphrase_precision(keys, ref_keys)
    r = keyphrase_recall(keys, ref_keys)

    if p == 0:
        return 0

    if r == 0:
        return 0

    # return 1.0 / ((alpha / p) + ((1 - alpha) / r))
    f1 = (2.0 * p * r) / (p + r)
    return f1


def keyphrase_precision(keys, ref_keys):
    score = 0
    for key in keys:
        if key in ref_keys:
            score += 1
        else:
            max_score = 0
            for ref_key in ref_keys:
                sc = precision(set(ref_key.split()), set(key.split()))
                if sc > max_score:
                    max_score = sc
            score += max_score

    if len(keys) == 0:
        return 0

    return score / len(keys)


def keyphrase_recall(keys, ref_keys):
    score = 0
    for ref_key in ref_keys:
        if ref_key in keys:
            score += 1
        else:
            max_score = 0
            for key in keys:
                sc = recall(set(ref_key.split()), set(key.split()))
                if sc > max_score:
                    max_score = sc
            score += max_score

    if len(ref_keys) == 0:
        return 0

    return score / len(ref_keys)


def keyword_score(extracted_keywords, reference_keywords):
    if extracted_keywords is None or reference_keywords is None:
        return None

    if len(extracted_keywords) == 0 or len(reference_keywords) == 0:
        return 0, 0, 0

    extracted_keywords_set = set(extracted_keywords)
    reference_keywords_set = set(reference_keywords)

    f_mes = f_measure(reference_keywords_set, extracted_keywords_set)
    prec = precision(reference_keywords_set, extracted_keywords_set)
    rec = recall(reference_keywords_set, extracted_keywords_set)

    return f_mes, prec, rec


def keyphrase_score(extracted_keyphrases, reference_keyphrases):
    f_mes = keyphrase_f1(extracted_keyphrases, reference_keyphrases)
    prec = keyphrase_precision(extracted_keyphrases, reference_keyphrases)
    rec = keyphrase_recall(extracted_keyphrases, reference_keyphrases)

    return f_mes, prec, rec


def get_reference_keywords(reference_text):
    text_tokens = clean_and_tokenize(reference_text)
    reference_keywords = [token.text for token in text_tokens]

    return reference_keywords


def get_reference_ngrams(reference_text):
    """Assumes reference keys are ngrams one on each line"""
    reference_text = str(reference_text)
    ngrams = reference_text.splitlines()
    return ngrams


def format_result(scores):
    f1, precision, recall = scores

    f1 = format(f1, '.4f')
    precision = format(precision, '.4f')
    recall = format(recall, '.4f')

    result = ("F1: {}, Precision: {}, Recall: {}".format(
        f1, precision, recall, ).replace(", ", "\n"))
    return result


def aggregate_results():
    os.chdir(KEYWORD_DIR)
    n = 0
    av_lem_w_f1 = av_lem_w_p = av_lem_w_r = 0
    av_lem_kp_f1 = av_lem_kp_p = av_lem_kp_r = 0

    av_w_f1 = av_w_p = av_w_r = 0
    av_kp_f1 = av_kp_p = av_kp_r = 0

    av_filt_w_f1 = av_filt_w_p = av_filt_w_r = 0
    av_filt_kp_f1 = av_filt_kp_p = av_filt_kp_r = 0

    av_est_w_f1 = av_est_w_p = av_est_w_r = 0
    av_est_kp_f1 = av_est_kp_p = av_est_kp_r = 0

    for file in glob.glob("*.abstr"):
        input_path = os.path.join(KEYWORD_DIR, file)
        reference_path = os.path.join(KEYWORD_DIR, file.replace(".abstr", ".key"))

        file_text = read_file(input_path)
        reference_text = read_file(reference_path)

        document = preprocessing.clean_and_tokenize(file_text)
        keyword_provider = KeywordProvider(document)

        ref_keywords = get_reference_keywords(reference_text)
        ref_ngrams = get_reference_ngrams(reference_text)

        keyphrases_TR = get_graph_keywords_2(file_text)
        keyphrases_TR_lem = keyword_provider.show_key_phrases(trim=True)

        keyphrases_filtered = get_graph_keywords_4(file_text, filter=True)

        # Get the baselines keywords
        baseline_keywords = get_baseline_keywords(file_text, len(ref_keywords))
        established_keyphrases = gensim_keywords(file_text).split('\n')

        k_TR_lem = list(set(' '.join(keyphrases_TR_lem).split()))
        (f1, p, r) = keyword_score(k_TR_lem, ref_keywords)
        av_lem_w_f1 += f1
        av_lem_w_p += p
        av_lem_w_r += r

        (f1, p, r) = keyphrase_score(keyphrases_TR_lem, ref_ngrams)
        av_lem_kp_f1 += f1
        av_lem_kp_p += p
        av_lem_kp_r += r

        k_TR = list(set(' '.join(keyphrases_TR).split()))
        (f1, p, r) = keyword_score(k_TR, ref_keywords)
        av_w_f1 += f1
        av_w_p += p
        av_w_r += r

        (f1, p, r) = keyphrase_score(k_TR, ref_ngrams)
        av_kp_f1 += f1
        av_kp_p += p
        av_kp_r += r

        k_TR_filt = list(set(' '.join(keyphrases_filtered).split()))
        (f1, p, r) = keyword_score(k_TR_filt, ref_keywords)
        av_filt_w_f1 += f1
        av_filt_w_p += p
        av_filt_w_r += r

        (f1, p, r) = keyphrase_score(k_TR_filt, ref_ngrams)
        av_filt_kp_f1 += f1
        av_filt_kp_p += p
        av_filt_kp_r += r

        k_established = list(set((' '.join(established_keyphrases)).split()))
        (f1, p, r) = keyword_score(k_established, ref_keywords)
        av_est_w_f1 += f1
        av_est_w_p += p
        av_est_w_r += r

        (f1, p, r) = keyphrase_score(k_established, ref_ngrams)
        av_est_kp_f1 += f1
        av_est_kp_p += p
        av_est_kp_r += r

        n = n+1
        
        break

    av_est_w_f1 *= 100 / n
    av_est_kp_f1 *= 100 / n

    av_est_w_p *= 100 / n
    av_est_kp_p *= 100 / n

    av_est_w_r *= 100 / n
    av_est_kp_r *= 100 / n

    av_lem_w_f1 *= 100 / n
    av_lem_kp_f1 *= 100 / n

    av_lem_w_p *= 100 / n
    av_lem_kp_p *= 100 / n

    av_lem_w_r *= 100 / n
    av_lem_kp_r *= 100 / n

    av_w_f1 *= 100 / n
    av_kp_f1 *= 100 / n

    av_w_p *= 100 / n
    av_kp_p *= 100 / n

    av_w_r *= 100 / n
    av_kp_r *= 100 / n

    av_filt_w_f1 *= 100 / n
    av_filt_kp_f1 *= 100 / n

    av_filt_w_p *= 100 / n
    av_filt_kp_p *= 100 / n

    av_filt_w_r *= 100 / n
    av_filt_kp_r *= 100 / n

    res_kp = "F1 lemmas:{} , Precision lemmas:{}, Recall lemmas:{} \n \
    F1 w/o lemmas:{} , Precision w/o lemmas:{}, Recall w/o lemmas:{} \n \
    F1 filtered:{} , Precision filtered:{}, Recall filtered:{} \n \
    F1 established:{} , Precision established:{}, Recall established:{} \n".\
        format(av_lem_kp_f1, av_lem_kp_p, av_lem_kp_r,
               av_kp_f1, av_kp_p, av_kp_r,
               av_filt_kp_f1, av_filt_kp_p, av_filt_kp_r,
               av_est_kp_f1,av_est_kp_p, av_est_kp_r)

    res_w = "F1 lemmas:{} , Precision lemmas:{}, Recall lemmas:{} \n \
    F1 w/o lemmas:{} , Precision w/o lemmas:{}, Recall w/o lemmas:{} \n \
    F1 filtered:{} , Precision filtered:{}, Recall filtered:{} \n \
    F1 established:{} , Precision established:{}, Recall established:{} \n". \
        format(av_lem_w_f1, av_lem_w_p, av_lem_w_r,
               av_w_f1, av_w_p, av_w_r,
               av_filt_w_f1, av_filt_w_p, av_filt_w_r,
               av_est_w_f1, av_est_w_p, av_est_w_r)

    output_path_w = os.path.join(KEYWORD_DIR, "av_res_w_2.txt")
    output_path_kp = os.path.join(KEYWORD_DIR, "av_res_kp_2.txt")

    print_to_file(res_w, output_path_w)
    print_to_file(res_kp, output_path_kp)


if __name__ == '__main__':
    aggregate_results()
    # FILE_PATH = input('Enter a file path to get keywords for: ')
    # REFERENCE_PATH = input('Enter a file path for reference keywords: ')
    #
    # FILE_TEXT = read_file(FILE_PATH)
    # document = preprocessing.clean_and_tokenize(FILE_TEXT)
    # keyword_provider = KeywordProvider(document)
    #
    # REFERENCE_TEXT = read_file(REFERENCE_PATH)
    # REFERENCE_KEYWORDS = get_reference_keywords(REFERENCE_TEXT)
    # REFERENCE_NGRAMS = get_reference_ngrams(REFERENCE_TEXT)
    #
    # # Get keyphrases from each implementation
    # graph_keyphrases = get_graph_keywords(FILE_TEXT)
    # keyphrases_TR = get_graph_keywords_2(FILE_TEXT)
    # keyphrases_TR_lem = keyword_provider.show_key_phrases(trim=True)
    #
    # keyphrases_filtered = get_graph_keywords_4(FILE_TEXT, filter=True)
    #
    # # Get the baselines keywords
    # baseline_keywords = get_baseline_keywords(FILE_TEXT, len(REFERENCE_KEYWORDS))
    # established_keyphrases = gensim_keywords(FILE_TEXT).split('\n')
    #
    # # Print out the scores
    # print('\nMy TR implementation with lemmas:')
    # keywords = list(set(' '.join(keyphrases_TR_lem).split()))
    # print(keyphrases_TR_lem)
    # print(('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS))))
    # print(('Per-keyphrase: \n' + str(keyphrase_score(keyphrases_TR_lem, REFERENCE_NGRAMS))))
    #
    # print('\nMy TR implementation w/o lemmas:')
    # keywords = list(set(' '.join(keyphrases_TR).split()))
    # print(('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS))))
    # print(('Per-keyphrase: \n' + str(keyphrase_score(keyphrases_TR, REFERENCE_NGRAMS))))
    #
    # print('\nMy TR implementation w/o lemmas AND filtered:')
    # keywords = list(set(' '.join(keyphrases_filtered).split()))
    # print(('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS))))
    # print(('Per-keyphrase: \n' + str(keyphrase_score(keyphrases_filtered, REFERENCE_NGRAMS))))
    #
    # print('\nEstablished implementation:')
    # keywords = list(set((' '.join(established_keyphrases)).split()))
    # print(established_keyphrases)
    # print('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS)))
    # print('Per-keyphrase: \n' + str(keyphrase_score(established_keyphrases, REFERENCE_NGRAMS)))
    #
    # print('\nBaseline implementation:')
    # print((keyword_score(baseline_keywords, REFERENCE_KEYWORDS)))

