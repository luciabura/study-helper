#!/usr/bin/env python

import sys
import os.path
import subprocess

debug = '2>/dev/null'

HEILMAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), './qg-heilman')


def generate_questions_from_path(text=None, article_path=None, nquestions=None):
    os.chdir(HEILMAN_DIR)
    if article_path:
        cmd = './scripts/run.sh question-asker --debug %s < %s' % (debug, article_path)
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")

    elif text:
        # new_text = "'"+text+"'"
        cmd = './scripts/run.sh question-asker --debug %s' % debug
        process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        output, error = process.communicate(bytes(text.encode("utf-8")))
        output = output.decode("utf-8")
        process.terminate()
    else:
        return

    questions = []

    for line in output.splitlines():
      if line.strip() == '':
        continue

      question, sentence, answer, score = line.strip().split('\t')
      questions.append(question)

    # print('\n'.join(questions[0:min(len(questions), nquestions)]))
    return questions


if __name__ == '__main__':
    article_path = input("Path:")
    questions = 5
    # generate_questions_from_path(article_path=article_path, nquestions=questions)
    generate_questions_from_path(text="This is a bit of text.", nquestions=questions)

