from baselines.keyword_baseline import get_keywords as get_baseline_keywords
from keyword_extraction.keywords import get_keywords as get_graph_keywords
from keyword_extraction.keywords_TR import get_keywords as get_graph_keywords_2
from keyword_extraction.keywords_TR_lem import get_keywords as get_graph_keywords_3
from keyword_extraction.keywords_filtered import get_keywords as get_graph_keywords_4
from nltk.metrics.scores import f_measure, precision, recall
from utilities.read_write import read_file
from text_processing.preprocessing import clean_and_tokenize
from gensim.summarization import keywords as gensim_keywords


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
    if p is 0 or r is 0:
        return 0

    return 1.0 / (alpha / p + (1 - alpha) / r)


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


if __name__ == '__main__':
    FILE_PATH = input('Enter a file path to get keywords for: ')
    REFERENCE_PATH = input('Enter a file path for reference keywords: ')

    FILE_TEXT = read_file(FILE_PATH)
    REFERENCE_TEXT = read_file(REFERENCE_PATH)
    REFERENCE_KEYWORDS = get_reference_keywords(REFERENCE_TEXT)
    REFERENCE_NGRAMS = get_reference_ngrams(REFERENCE_TEXT)

    # Get keyphrases from each implementation
    graph_keyphrases = get_graph_keywords(FILE_TEXT)
    keyphrases_TR = get_graph_keywords_2(FILE_TEXT)
    keyphrases_TR_lem = get_graph_keywords_3(FILE_TEXT)
    keyphrases_filtered = get_graph_keywords_4(FILE_TEXT, filter=True)

    # Get the baselines keywords
    baseline_keywords = get_baseline_keywords(FILE_TEXT, len(REFERENCE_KEYWORDS))
    established_keyphrases = gensim_keywords(FILE_TEXT).split('\n')

    # Print out the scores
    print('\nMy TR implementation with lemmas:')
    keywords = list(set(' '.join(keyphrases_TR_lem).split()))
    print(('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS))))
    print(('Per-keyphrase: \n' + str(keyphrase_score(keyphrases_TR_lem, REFERENCE_NGRAMS))))

    print('\nMy TR implementation w/o lemmas:')
    keywords = list(set(' '.join(keyphrases_TR).split()))
    print(('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS))))
    print(('Per-keyphrase: \n' + str(keyphrase_score(keyphrases_TR, REFERENCE_NGRAMS))))

    print('\nMy TR implementation w/o lemmas AND filtered:')
    keywords = list(set(' '.join(keyphrases_filtered).split()))
    print(('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS))))
    print(('Per-keyphrase: \n' + str(keyphrase_score(keyphrases_filtered, REFERENCE_NGRAMS))))

    print('\nEstablished implementation:')
    keywords = list(set((' '.join(established_keyphrases)).split()))
    print('Per-word: \n' + str(keyword_score(keywords, REFERENCE_KEYWORDS)))
    print('Per-keyphrase: \n' + str(keyphrase_score(established_keyphrases, REFERENCE_NGRAMS)))

    print('\nBaseline implementation:')
    print((keyword_score(baseline_keywords, REFERENCE_KEYWORDS)))

