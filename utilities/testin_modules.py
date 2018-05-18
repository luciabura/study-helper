import pickle
import nltk.translate.bleu_score as NLTK_BLEU

from question_generation.sentence_simplifier import get_coreferences
from utilities.read_write import read_file


def aggregate_result(text):
    lines = text.splitlines()
    it = iter(lines)
    count = m = h = 0
    for line in it:
        if not line:
            break

        m += float(line.split(':')[1])

        line = next(it)

        if m > 0:
            count += 1
            h += float(line.split(':')[1])

    print(m/count, h/count)




if __name__ == '__main__':
    # # f = open('ngram_model.pickle', 'rb')
    # f = open('ngram_counter.pickle', 'rb')
    # # model = pickle.load(f)
    # counter = pickle.load(f)
    # f.close()
    # model =
    # print(model.perplexity("What is love?"))
    # reference = "This is a sentence. This is another sentence."
    # summary = ""
    # bleu = NLTK_BLEU.sentence_bleu([reference], summary)
    # print(bleu)
    text = read_file(input("File:"))
    aggregate_result(text)

