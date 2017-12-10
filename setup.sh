#!/usr/bin/env bash
PYTHON=`which python`


## Install pip in case it wasn't installed 
echo "Installing pip"
sudo apt get install pip 
pip install --upgrade pip


## Install dependencies
echo "Instaling dependencies"
pip install -r requirements.txt
sudo $PYTHON setup.py develop


## Download nltk data 
sudo $PYTHON -c "import nltk; nltk.download('stopwords');nltk.download('punkt');nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
sudo $PYTHON -m spacy download en
