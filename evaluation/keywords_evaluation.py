from baselines.keyword_baseline import get_keywords as get_baseline_keywords
from keyword_extraction.keywords import get_keywords as get_graph_keywords
from keyword_extraction.keywords_TR import get_keywords as get_graph_keywords_2
from keyword_extraction.keywords_TR_lem import get_keywords as get_graph_keywords_3
from nltk.metrics.scores import f_measure, precision, recall
from utilities.utils import read_file
from preprocessing.preprocessing import clean_and_tokenize
from summa import keywords as summa_keywords


def keyword_score(extracted_keywords, reference_keywords):
    extracted_keywords_set = set(extracted_keywords)
    reference_keywords_set = set(reference_keywords)

    f_mes = f_measure(reference_keywords_set, extracted_keywords_set)
    prec = precision(reference_keywords_set, extracted_keywords_set)
    rec = recall(reference_keywords_set, extracted_keywords_set)

    f_mes = format(f_mes, '.4f')
    prec = format(prec, '.4f')
    rec = format(rec, '.4f')

    return f_mes, rec, prec


def keyphrase_score(extracted_keyphrases, reference_keyphrases):
    extracted_keyphrase_set = set(extracted_keyphrases)
    reference_keyphrase_set = set(reference_keyphrases)

    f_mes = f_measure(reference_keyphrase_set, extracted_keyphrase_set)
    prec = precision(reference_keyphrase_set, extracted_keyphrase_set)
    rec = recall(reference_keyphrase_set, extracted_keyphrase_set)

    f_mes = format(f_mes, '.4f')
    prec = format(prec, '.4f')
    rec = format(rec, '.4f')

    return f_mes, rec, prec

def get_reference_keywords(reference_text):
    text_tokens = clean_and_tokenize(reference_text)
    reference_keywords = [token.text for token in text_tokens]

    return reference_keywords


def get_reference_ngrams(reference_text):
    """Assumes reference keys are ngrams one on each line"""
    reference_text = unicode(reference_text)
    ngrams = reference_text.splitlines()
    return ngrams


if __name__ == '__main__':
    FILE_PATH = raw_input('Enter a file path to get keywords for: ')
    REFERENCE_PATH = raw_input('Enter a file path for reference keywords: ')

    FILE_TEXT = read_file(FILE_PATH)
    REFERENCE_TEXT = read_file(REFERENCE_PATH)
    REFERENCE_KEYWORDS = get_reference_keywords(REFERENCE_TEXT)
    REFERENCE_NGRAMS = get_reference_ngrams(REFERENCE_TEXT)

    # Get keyphrases from each implementation
    graph_keyphrases = get_graph_keywords(FILE_TEXT)
    keyphrases_TR = get_graph_keywords_2(FILE_TEXT)
    keyphrases_TR_lem = get_graph_keywords_3(FILE_TEXT)

    # Get the baselines keywords
    baseline_keywords = get_baseline_keywords(FILE_TEXT, len(REFERENCE_KEYWORDS))
    established_keyphrases = (summa_keywords.keywords(FILE_TEXT)).split('\n')

    # Print out the scores
    print '\nMy TR implementation with lemmas:'
    keywords = list(set(' '.join(keyphrases_TR_lem).split()))
    print 'Per-word:' + str(keyword_score(keywords, REFERENCE_KEYWORDS))
    print 'Per-keyphrase:' + str(keyphrase_score(keyphrases_TR_lem, REFERENCE_NGRAMS))

    print '\nMy TR implementation w/o lemmas:'
    keywords = list(set(' '.join(keyphrases_TR).split()))
    print 'Per-word:' + str(keyword_score(keywords, REFERENCE_KEYWORDS))
    print 'Per-keyphrase:' + str(keyphrase_score(keyphrases_TR, REFERENCE_NGRAMS))

    print '\nEstablished implementation:'
    keywords = list(set((' '.join(established_keyphrases)).split()))
    print 'Per-word:' + str(keyword_score(keywords, REFERENCE_KEYWORDS))
    print 'Per-keyphrase:' + str(keyphrase_score(established_keyphrases, REFERENCE_NGRAMS))

    print '\nBaseline implementation:'
    print keyword_score(baseline_keywords, REFERENCE_KEYWORDS)

