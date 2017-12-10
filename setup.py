"""A setuptools based setup module.
Inspired from https://github.com/pypa/sampleproject/blob/master/setup.py
"""

from setuptools import setup, find_packages

setup(
    name='Study Helper',
    version='0.0.1',

    description='A study helper based on natural language processing',
    url='https://github.com/luciabura/study-helper',

    author='Lucia Bura',
    author_email='luciabura@gmail.com',

    keywords=['NLP', 'keyphrase extraction', 'summarization', 'question generation'],

    packages=find_packages(),
    install_requires=['networkx>=2.0', 'nltk>=3.2.1', 'numpy>=1.13.3', 'spacy'],
)
