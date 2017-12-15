import codecs
from utilities.utils import read_file
from autocorrect import spell
import glob, os
import preprocessing.preprocessing as preprocessor


def clean_files(ref_name, text_words, ref_ngrams):
    file = codecs.open(ref_name, 'w', encoding='utf-8')
    new_ngrams = []
    for ngram in ref_ngrams:
        OK = 1
        for word in ngram:
            if word not in text_words:
                print(word)
                OK = 0
                break

        if(OK == 1):
            new_ngrams.append(ngram)

    for ngram in new_ngrams:
        keyword = ' '.join(ngram)
        print(keyword, file=file)


def clean_hulth():
    directory = input('Directory path: ')
    text_ending = '.abstr'
    key_ending = '.uncontr'

    os.chdir(directory)
    for file in glob.glob("*" + text_ending):
        name = file.split('.')[0]
        ref_name = name + key_ending

        text = read_file(file)
        ref = read_file(ref_name)

        text_words = [token.text.lower() for token in preprocessor.clean_and_tokenize(text)]

        ref_ngrams = ref.split(';')
        for i, ngram in enumerate(ref_ngrams):
            ref_ngrams[i] = preprocessor.clean_and_format(ngram).lower().split(' ')
            if '' in ref_ngrams[i]:
                ref_ngrams[i].remove('')

        ref_name = name+".key"
        clean_files(ref_name, text_words, ref_ngrams)


def clean_krapivin():
    # directory = raw_input('Directory path: ')
    # text_ending = '.txt'
    # key_ending = '.key'
    #
    # os.chdir(directory)
    # for file in glob.glob("*" + text_ending):
    #     name = file.split('.')[0]
    #     ref_name = name + key_ending
    #
    #     text = read_file(file)
    #     ref = read_file(ref_name)
    FILE_PATH = input('Enter the absolute path of '
                          'the file you want to extract the keywords from: \n')
    FILE_TEXT = read_file(FILE_PATH)
    lines = FILE_TEXT.splitlines()
    new_lines = []
    for line in lines:
        if len(line) > 3:
            new_lines.append(line)

    text = ' '.join(new_lines)
    text_words = [token.text for token in preprocessor.clean_and_tokenize(text)]
    text_words = correct_words(text_words)
    text = ' '.join(text_words)
    print(text)
    # print new_lines

def correct_words(words):
    new_words = []
    for word in words:
        if not word.isalpha():
            new_words.append(word)
        else:
            new_word = spell(word)
            new_words.append(new_word)

    return new_words

if __name__ == '__main__':
    # clean_hulth()
    clean_krapivin()
