# from pycorenlp import StanfordCoreNLP
# NLP = StanfordCoreNLP('http://localhost:9000')
import json
import requests

tregex_url = "http://localhost:9000/tregex"


if __name__ == '__main__':
    PATTERN = "NP < (NP=noun !$-- NP $+ (/,/ $++ NP|PP=appositive !$ CC|CONJP)) >> (ROOT <<# /^VB.*/=mainverb) "
    patt = "{tag: SUBJ}"
    texxt = u"The meeting, in 1980, was important."
    # try:
    #     r = requests.get(
    #         endpoint, params={
    #             'pattern': PATTERN,
    #             'filter': False,
    #         }, data=texxt)
    #     r.raise_for_status()
    #     print(r.text)
    #     print(json.loads(r.text))
    # except requests.HTTPError as e:
    #     print('well fuck')
    # except json.JSONDecodeError:
    #     print('meh')

    r = requests.post(tregex_url, data=texxt, params={'pattern': PATTERN, 'filter': False})
    print(r.json())
