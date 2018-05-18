A Study Helper based on Natural Language Processing 
===
[![Build Status](https://travis-ci.org/luciabura/study-helper.svg?branch=master)](https://travis-ci.org/luciabura/study-helper)
*This repository has been set up for a Part II Project.* 
## Description
The aim of the project is to build an educational tool based on key–phrase extraction, summarization and question generation algorithms.

A more thorough description of the scope and intents of the project can be found 
in the project proposal under: ```study-helper/documents/proposal.pdf```

### System Overview

**Key-phrase and Sentence extraction**

Implemented in ``study-helper/keyword_extraction`` and ``study-helper/sentence_extraction`` packages, respectively, 
following an unsupervised graph-based learning model outlined by Radu Mihalcea and Paul Tarau in his paper: [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

**Question generation step**
Implemented in ``study-helper/question_generation``, inspired by the two papers: [Question Generation via Overgenerating
Transformations and Ranking](http://www.cs.cmu.edu/~ark/mheilman/questions/papers/heilman-smith-qg-tech-report.pdf) and [Automatic Question Generation: From NLU to NLG](https://link.springer.com/chapter/10.1007/978-3-319-39583-8_3)

## Getting up and running 

*Warning, this project uses Python 3.5!*

### Installing:

To make everything easy to install I've written a setup script which you can run
from the command line via: 

```bash
sudo bash ./setup.sh
```

*Observation: This will take some time, depending on what you already have installed on your machine*

Alternatively, you can install all the required dependencies individually: 

* [NLTK](https://www.nltk.org/install.html)
* [SpaCy](https://spacy.io/usage/)
* [Networkx](https://networkx.github.io/documentation/networkx-1.1/install.html)

For the python scripts to work properly, you will also need to 
download the required models and dictionaries to be loaded:

```bash
sudo python3 -m spacy download en_core_web_md
sudo python3 -c "import nltk; nltk.download('stopwords');nltk.download('punkt');nltk.download('wordnet');"
```

#### Keyphrase extraction

To inspect the behaviour of the files you can either run them individually, for example:

```bash
python3 keywords.py
```

or use them in your own files:

```python
from study-helper.keyword_extraction.keywords import get_keywords
```

**Examples to run against:**

I've also provided some example text and corresponding hand-annotated keywords 
that the program can be tested with. These are in the `data` directory

For getting the keywords, any text file should do, eg: upon being prompted
asking for filepath, you can provide a path to one of the `.abstr` files 

I've also made an evaluation step for the keyword extraction, which can be used 
with the ``.abstr`` and ``.key`` files respectively. *Preferably with matching names.*

#### Sentence extraction

Similar to the keyword extraction, to see the summarizer on its own simply run:


```bash
python3 sentence_provider.py
```