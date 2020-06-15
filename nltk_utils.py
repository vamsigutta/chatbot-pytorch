from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
ps = PorterStemmer()

## Tokenize
def tokenize(sentence):
    return word_tokenize(sentence)

## Stem
def stem(word):
    return ps.stem(word)


## Bag of Words
def bag_of_words(tokenized_sentences, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    tokenized_sentences = [stem(word) for word in tokenized_sentences]
    for index, word in enumerate(all_words):
        if word in tokenized_sentences:
            bag[index] = 1.0
    return bag
