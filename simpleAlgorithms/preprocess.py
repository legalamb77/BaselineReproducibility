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
from random import shuffle
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
Idea: Use this on both positive and negative files to get positive and negative test sets, then label them after translation
Input:
    file names for the unprocessed examples
    testSize: Integer size of the test set, in terms of examples
'''
def trainTestSplit(exFile, testSize):
    fl = open(exFile).readlines()
    shuffle(fl)
    testSet = fl[:testSize]
    trainSet = fl[testSize:]
    return trainSet, testSet

def countFreqs(w2i, pos, neg, x, y):
    count = 0
    for pEx, nEx in zip(pos, neg):
        posLine = pEx.split(' ')
        for token in posLine:
            if token in w2i:
                x[count][w2i[token]] += 1
            else:
                pass
        y[count] = 1
        count+=1
        negLine = nEx.split(' ')
        for token in negLine:
            if token in w2i:
                x[count][w2i[token]] += 1
            else:
                pass
        y[count] = 0
        count += 1
    return x, y

def translateExamplesRT(w2i, trainingPos, trainingNeg, testPos, testNeg):
    # Initialize numpy arrays for training and test
    training_x = np.zeros((len(trainingPos)+len(trainingNeg), len(w2i)))
    training_y = np.zeros(len(trainingPos)+len(trainingNeg))
    test_x = np.zeros((len(testPos)+len(testNeg), len(w2i)))
    test_y = np.zeros(len(testPos)+len(testNeg))
    count = 0
    # Fill out the training arrays with frequencies
    training_x, training_y = countFreqs(w2i, trainingPos, trainingNeg, training_x, training_y)
    test_x, test_y = countFreqs(w2i, testPos, testNeg, test_x, test_y)
    return training_x, training_y, test_x, test_y

if __name__ == '__main__':
    pass
