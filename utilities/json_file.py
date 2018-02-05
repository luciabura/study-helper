import codecs
import json

import os

from utilities.read_write import print_to_file

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
questions_filepath = os.path.join(__location__, 'questions_corpus.txt')

json_filepath = input('JSON Filepath:')

with open(json_filepath) as f:
    data_file = json.load(f)


def get_text_with_questions():
    text_with_questions = {}

    data = data_file['data']
    for data_el in data:
        title = data_el['title']
        for paragraph in data_el['paragraphs']:
            text = paragraph['context']
            questions = []
            for qa in paragraph['qas']:
                questions.append(qa['question'])

            text_with_questions[text] = questions

    return text_with_questions


def print_all_questions():
    text_with_questions = get_text_with_questions()
    all_questions = []
    for questions in text_with_questions.values():
        all_questions.extend(questions)

    text = "\n".join(all_questions)
    print_to_file(text, questions_filepath)


if __name__ == '__main__':
    print_all_questions()





