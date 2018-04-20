'''
@author Theodore Morley
Goals:
    -Separate into train and test sets
        -Paper uses 10% validation, 10% test, 80% training for RT
        -So just seperate 10% from pos and neg into a test set
    -Optional: If specified, remove stops and punctuation
    -Make list of remaining training words
    -Use word2Ix approach to specify vector length
    -Just ignore words that are not in the training data
    -Next translate each of the documents into np arrays using the w2ix
'''
from nltk.corpus import stopwords
import string
stops = set(stopwords.words('english'))
punct = string.punctuation

'''
Input:
    training_vocab: list containing all unique words in the document
'''
def word2ix(training_vocab, rm_stops, rm_punct):
    count = 0
    w2x = dict()
    for word in training_vocab:
        if word in stops and rm_stops:
            pass
        elif word in punct and rm_punct:
            pass
        else:
            w2x[word] = count
            count += 1

if __name__ == '__main__':
    pass
