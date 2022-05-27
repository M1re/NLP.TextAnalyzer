
# Press the green button in the gutter to run the script.
from sklearn.feature_extraction.text import CountVectorizer

import ONE_Encoder
from EN_Analyzer import EN_Analyzer
from UA_Analyzer import UA_Analyzer

if __name__ == '__main__':
    one = ONE_Encoder
    print("\nUkrainian Text")
    with open('Ukrainian.txt') as textio:
        text = ''.join(textio.readlines())
        with open('UA_Stopwords.txt') as words:
            tokens, lemmas, stems = UA_Analyzer(text)
        print("Tokens: {}".format(tokens))
        print("Lemmas: {}".format(lemmas))
        print("Stems: {}".format(stems))


    print("\nEnglish Text")
    with open('English.txt') as textio:
        text = ''.join(textio.readlines())
        tokens, lemmas, stems = EN_Analyzer(text)
        print("Tokens: {}".format(tokens))
        print("Lemmas: {}".format(lemmas))
        print("Stems: {}".format(stems))
        print("ONE: \n{}".format(ONE_Encoder.ONE(lemmas[:10])))

        print("\nBOW Text")
        vectorizer = CountVectorizer(lowercase=True, stop_words='english')
        vectorizer = vectorizer.fit(tokens[:10])
        print(vectorizer.vocabulary_)
        vector = vectorizer.transform(tokens)
        print("Vector shape: {}".format(vector.shape))
        print("Vector to array: \n{}".format(vector.toarray()[:10]))



