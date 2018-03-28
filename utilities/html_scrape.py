from bs4 import BeautifulSoup
from urllib.request import urlopen
import question_generation.questions_2 as QG

quote_page = "http://www.cl.cam.ac.uk/teaching/1718/FJava/workbook1.html"

# query the website and return the html to the variable ‘page’
page = urlopen(quote_page)
soup = BeautifulSoup(page, 'html.parser')
page_text = soup.get_text()

questions = QG.generate_questions_trial(text=page_text)
for question in questions:
        # perplexity = spacy_perplexity(question.content)
        # print("Q: {}\nPerplexity: {}\n".format(question.content, perplexity))
        print("Q: {}".format(question.content))
