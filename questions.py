import argparse

from question_generation.question_provider import QuestionProvider
from utilities.html_scrape import get_html_text
from utilities.read_write import print_to_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-sents", "--sentences",
                        help="Show only the relevant sentences. Not available for short texts.")

    parser.add_argument("-keys", "--keywords",
                        help="Show only the text keywords. Not avaiable for short texts.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show all information about the questions generated including originating sentence and "
                             "enclosed keywords.")

    parser.add_argument("-t", "--topic", type=str, default=None,
                        help="Add a word describing the desired topic.")

    inputs = parser.add_mutually_exclusive_group()

    inputs.add_argument("-url", type=str, default=None,
                        help="URL to page you want to analyze.")

    inputs.add_argument('-file', type=argparse.FileType('r'), default=None,
                        help="Absolute path to file you want to analyze.")

    parser.add_argument("output_file", nargs='?', type=argparse.FileType('w'), default=None)

    parser.usage = "questions.py [-sents] [-keys] [-v] [-t topic_word] [-url url_path] [-file file_path] [output_file]"

    opts = parser.parse_args()

    return opts


if __name__ == '__main__':
    opts = parse_args()
    if opts.file:
        text = opts.file.read()
    elif opts.url:
        text = get_html_text(opts.url)
    else:
        text = input("Please enter the text to be processed:\n")

    if opts.topic:
        question_provider = QuestionProvider(text, topic=opts.topic)
    question_provider = QuestionProvider(text)
    if opts.sentences:
        question_provider.sents_only = True
    if opts.keywords:
        question_provider.keys_only = True

    if not opts.output_file:
        question_provider.print_questions(verbose=opts.verbose)
    else:
        questions = question_provider.questions
        all_qs = [question.text for question in questions]
        all_qs = '\n'.join(all_qs)
        print_to_file(all_qs, opts.output_file)
