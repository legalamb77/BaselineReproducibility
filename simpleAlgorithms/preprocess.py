'''
@author Theodore Morley
Goals:
    -Separate into train and test sets
        -Paper uses 10% validation, 10% test, 80% training for RT
        -So just seperate 10% from pos and neg into a test set
    #-Optional: If specified, remove stops and punctuation
    -Make list of remaining training words
    #-Use word2Ix approach to specify vector length
    #-Just ignore words that are not in the training data
    -Next translate each of the documents into np arrays using the w2ix
'''
from nltk.corpus import stopwords
import string
import numpy as np

stops = set(stopwords.words('english'))
punct = string.punctuation

'''
Input:
    training_vocab: list containing all unique words in the document
Output:
    Dictionary mapping from a word to its index in the vector representation
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
    return w2x

'''
Input:
    trainingPost and trainingNeg: Lists containing all the training examples (unprocessed) of positive and negative
Output:
    set containing unique words in the training set
'''
def getVocab(trainingPos, trainingNeg):
    strPos = ' '.join(trainingPos)
    strNeg = ' '.join(trainingNeg)
    posUniqueWords = set(strPos.split(' '))
    negUniqueWords = set(negUniqueWords.split(' '))
    return posUniqueWords.union(negUniqueWords)

'''
Input:
    file names for the unprocessed positive and negative examples
'''
def trainTestSplit(posFile, negFile):
    pass

def translateExamples(w2i, trainingPos, trainingNeg, testPos, testNeg):
    pass

if __name__ == '__main__':
    pass
