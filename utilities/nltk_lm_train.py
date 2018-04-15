import nltk
from nltk.model import MLENgramModel, LaplaceNgramModel
from nltk.model import build_vocabulary
from nltk.model import count_ngrams
import pickle
from utilities.read_write import read_file
import text_processing.preprocessing as preprocess

if __name__ == '__main__':
    filepath = input("Training input for Ngram model:")
    text = read_file(filepath)
    sentences = preprocess.nltk_sentence_tokenize(text)
    docs = []
    for sentence in sentences:
        sent_comp = nltk.word_tokenize(sentence)
        docs.append(sent_comp)

    print(docs[0:5])
    docs_small = docs[0:5]

    # vocab = build_vocabulary(1, *docs_small)
    vocab = build_vocabulary(1, *docs)
    # counter = count_ngrams(3, vocab, *docs_small)
    counter = count_ngrams(3, vocab, *docs)
    model = LaplaceNgramModel(counter)

    # f = open('ngram_counter.pickle', 'wb')
    # pickle.dump(counter, f)
    # f.close()

    print(model.perplexity(nltk.word_tokenize("What is the abominable disease?")))
    # print(model.perplexity("What"))
