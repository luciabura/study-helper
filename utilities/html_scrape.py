from bs4 import BeautifulSoup
from urllib.request import urlopen
import question_generation.questions_2 as QG

quote_page = "http://www.cl.cam.ac.uk/teaching/1718/FJava/workbook1.html"
quote_page_2 = "https://realworldocaml.org/v1/en/html/memory-representation-of-values.html"
quote_page_3 = "https://en.wikipedia.org/wiki/Computer_network"

# query the website and return the html to the variable ‘page’
page = urlopen(quote_page_2)
soup = BeautifulSoup(page, 'html.parser')
page_text = soup.get_text()

full_text = []
all_ps = soup.find_all('p')
for p in all_ps:
    full_text.append(p.get_text())

full_text = '\n'.join(full_text)

questions = QG.generate_questions_trial(text=full_text)
for question in questions:
    # perplexity = spacy_perplexity(question.content)
    # print("Q: {}\nPerplexity: {}\n".format(question.content, perplexity))
    print("Q: {}".format(question.content))
