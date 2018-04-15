import math
from nltk.corpus import wordnet
from wordfreq import zipf_frequency, word_frequency
# print(wordnet.synsets('cat'))


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

# print(1/(word_frequency('cat', lang='en')* 1e6))
# print(1/(word_frequency('dictionary', lang='en')* 1e6))
# print(zipf_frequency('dictionary', lang='en'))
# print(zipf_frequency('cat', lang='en'))
# print(zipf_frequency('horse', lang='en'))
# print(zipf_frequency('machinery', lang='en'))
# print(zipf_frequency('Microsoft', lang='en'))
