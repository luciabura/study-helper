import urllib.error
from urllib.request import urlopen

from bs4 import BeautifulSoup


# quote_page = "http://www.cl.cam.ac.uk/teaching/1718/FJava/workbook1.html"
# quote_page_2 = "https://realworldocaml.org/v1/en/html/memory-representation-of-values.html"
# quote_page_3 = "https://en.wikipedia.org/wiki/Computer_network"

def get_html_text(page_link):
    # query the website and return the html to the variable ‘page’
    try:
        page = urlopen(page_link)
    except urllib.error.HTTPError as e:
        print(e.code)
        return
    except urllib.error.URLError as e:
        print(e.args)

    soup = BeautifulSoup(page, 'html.parser')
    # page_text = soup.get_text()

    full_text = []
    all_ps = soup.find_all('p')
    for p in all_ps:
        full_text.append(p.get_text())

    full_text = '\n'.join(full_text)

    return full_text
