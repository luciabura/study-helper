import math

import nltk
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer_Text
from sumy.utils import get_stop_words


def get_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    stemmer = Stemmer("english")
    # summarizer = Summarizer(stemmer)
    # summarizer = Summarizer_Lex(stemmer)
    summarizer = Summarizer_Text(stemmer)
    summarizer.stop_words = get_stop_words("english")

    length = len(nltk.sent_tokenize(text))
    sentences = []
    for sentence in summarizer(parser.document, math.ceil(length*0.2)):
        sentences.append(sentence._text)

    summary = '\n'.join(sentences)
    return summary
