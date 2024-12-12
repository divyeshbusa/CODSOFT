# print("::::::::::::::::::::::::: nltk_utils.py is running ::::::::::::::::::::::::::::::::")
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    
def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    
    tokenized_sentence = [ lemmatize(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    # print("Tokenized sentence:", tokenized_sentence)
    # print("All words:", all_words)
    # print("Bag of Words:", bag)

    return bag

    


