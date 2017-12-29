A Study Helper based on Natural Language Processing 
===

*This repository has been set up for a Part II Project.* 
## Description
The aim of the project is to build an educational tool based on key–phrase extraction, summarization and question generation algorithms.

A more thorough description of the scope and intents of the project can be found 
in the project proposal under: ```study-helper/documents/proposal.pdf```

### System Overview

**Key-phrase extraction step**

Implemented within ``study-helper/keyword_extraction`` package, 
following an unsupervised graph-based learning model outlined by Radu Mihalcea and Paul Tarau in his paper: [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

**Sentence extraction step** (summarization)
[TBD]

**Question generation step**
[TBD]

## Getting up and running 

*Warning!: The project is still in its very early development stages*

### Installing:

To make everything easy to install I've written a setup script which you can run
from the command line via: 

```bash
sudo bash ./setup.sh
```

*Observation: This will take some time, depending on what you already have installed on your machine*

Alternatively, you can install all the required dependencies individually: 

* [NLTK]()
* [SpaCy]()
* [SciPy]()
* [NumPy]()
* [Networkx]()

For the python scripts to work properly, you will also need to 
download the required models and dictionaries to be loaded:

```bash
sudo python -m spacy download en
sudo python -c "import nltk; nltk.download('stopwords');nltk.download('punkt');nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

#### Keyphrase extraction

To inspect the behaviour of the files you can either run them individually, for example:

```bash
python keywords.py
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

