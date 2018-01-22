import codecs
import csv
import os

import wikipedia

import text_processing.preprocessing as preprocessor

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
cs = codecs.open(os.path.join(__location__, 'CS_WORDS.txt'), encoding='utf-8')


def nips_words():
    nips_location = os.path.join(__location__, 'NIPS_WORDS.csv')
    words = []
    with open(nips_location, 'rb') as nips:
        reader = csv.reader(nips)
        for row in reader:
            words.append(row[0])

    return words

# CS_WORDS = preprocessor.word_tokenize(cs.read())
# NIPS_WORDS = nips_words()


def words_from_page(page):
    asc = to_ascii(page.content)
    words = preprocessor.nltk_word_tokenize(asc)
    return words


def wiki_words(page, depth=1):
    wiki = []

    page_words = words_from_page(page)
    wiki.extend(page_words)

    links = page.links
    links = [to_ascii(link) for link in links]

    link_text = ' '.join(links)
    link_words = preprocessor.nltk_word_tokenize(link_text)

    wiki.extend(link_words)

    if depth != 0:
        depth -= 1
        for link in links:
            try:
                print('Here')
                page = wikipedia.page(link)
                page_words = wiki_words(page, depth)
                wiki.extend(page_words)
            except:
                pass

    return wiki


def to_ascii(text):
    return text.encode('ascii', 'ignore')


def get_cs_words():
    text = cs.read()
    cs_words = text.splitlines()
    return cs_words


if __name__ == '__main__':
    page = wikipedia.page("Computer Science")
    # words = wiki_words(page, 1)
    # tokens = preprocessor.word_tokenize_2(text)
    # words = [token.text.lower() for token in tokens]
    # words = list(set(words))
    # words = [word for word in words if len(word) > 2]
    # for word in words:
    #     print>>file, word
    # cs_words = get_cs_words()
    # cs_words.extend(NIPS_WORDS)
    # words = sorted(list(set(cs_words)))
    # file = codecs.open('CS_WORDS.txt', 'w', encoding='utf-8')
    # for word in words:
    #     print>>file, word
