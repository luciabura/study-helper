#!/usr/bin/env bash
PYTHON=`which python`


## Install pip in case it wasn't installed 
echo "Installing pip"
sudo apt-get install python-pip 
sudo pip install --upgrade pip


## Install dependencies for main part
echo "Instaling main dependencies"
sudo pip install -r requirements.txt
sudo $PYTHON setup.py develop


## Download nltk data 
sudo $PYTHON -c "import nltk; nltk.download('stopwords');nltk.download('punkt');nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
sudo $PYTHON -m spacy download en


## Install dependencies for evaluation
echo "Instaling evaluation dependencies"
sudo pip install summa 
sudo pip install scipy 

echo "Completed!"
