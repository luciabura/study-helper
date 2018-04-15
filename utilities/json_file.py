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


def show_json_data():
    data = data_file['data']

    for data_el in data:
        question = data_el['question']
        answer = data_el['answer']
        print("Question:{}\nAnswer:{}\n".format(question, answer))


def get_text_with_questions_ms():
    new_data = {"data": []}

    for data_el in data_file:
        question = data_el["query"]+"?"
        answer = data_el["wellFormedAnswers"][0]
        question_and_answer = {'question':question, 'answer':answer}
        new_data['data'].append(question_and_answer)

    # with open('questions_and_answers.json', 'w') as f_json:
    #     json.dump(new_data, f_json)


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


def map_paragraphs_to_questions_json():
    new_data = {"data": []}

    data = data_file['data']
    for data_el in data:
        title = data_el['title']
        for paragraph in data_el['paragraphs']:
            text = paragraph['context']
            questions = []
            for qa in paragraph['qas']:
                questions.append(qa['question'])

            text_with_q = {'title': title, 'text': text, 'questions': questions}
            new_data['data'].append(text_with_q)

    with open('paragraphs_with_questions.json', 'w') as f_json:
        json.dump(new_data, f_json)


def print_all_questions():
    text_with_questions = get_text_with_questions()
    all_questions = []
    for questions in text_with_questions.values():
        all_questions.extend(questions)

    text = "\n".join(all_questions)
    print_to_file(text, questions_filepath)


if __name__ == '__main__':
    # print_all_questions()
    # map_paragraphs_to_questions_json()
    # get_text_with_questions_ms()
    show_json_data()



