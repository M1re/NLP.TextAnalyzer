import string

from nltk import word_tokenize
from pymorphy2 import MorphAnalyzer
from Uk_Stemmer.uk_stemmer import UkStemmer


def UA_Analyzer(text):
    Stemmer = UkStemmer()
    Lemmatizer = MorphAnalyzer()

    with open('UA_Stopwords.txt') as sw:
        stopwords = sw.readline().split(' ') + list(string.punctuation)

    tokens = [token
              for token in word_tokenize(text)
              if not token.lower() in stopwords]

    lemmas = [Lemmatizer.parse(token)[0].normal_form
              for token in tokens]

    stems = [Stemmer.stemWord(token)
             for token in tokens]

    return tokens, lemmas, stems
