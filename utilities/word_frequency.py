import math

from wordfreq import word_frequency


def sequence_surprize(text):
    word_list = text.split()
    av_s = 0
    for word in word_list:
        av_s += 1/(word_frequency(word, lang='en')*1e6)

    if len(word_list) > 1:
        av_s /= math.log(len(word_list))

    return av_s


text = ["velocity", "supervised learning", "idea", "value"]

for t in text:
    print("Phrase:{}\nSurprize:{}\n".format(t, sequence_surprize(t)))
