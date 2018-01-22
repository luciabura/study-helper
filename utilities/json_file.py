import json
from pprint import pprint

json_filepath = input('JSON Filepath:')

with open(json_filepath) as f:
    d = json.load(f)
text_with_questions = {}


data = d['data']
for data_el in data:
    title = data_el['title']
    for paragraph in data_el['paragraphs']:
        text = paragraph['context']
        questions = []
        for qa in paragraph['qas']:
            questions.append(qa['question'])

        text_with_questions[text] = questions

for el in text_with_questions.keys():
    print(el)
    print(text_with_questions[el])


