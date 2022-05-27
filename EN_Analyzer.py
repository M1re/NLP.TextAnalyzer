import string
from nltk import SnowballStemmer, WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords


def EN_Analyzer(text):
    Stemmer = SnowballStemmer(language='english')
    Lemmatizer = WordNetLemmatizer()

    stop_words = list(stopwords.words('english')) + list(string.punctuation)
    tokens = [token
              for token in word_tokenize(text)
              if not token.lower() in stop_words]

    lemmas = [Lemmatizer.lemmatize(token)
              for token in tokens]

    stems = [Stemmer.stem(token)
             for token in tokens]

    return tokens, lemmas, stems