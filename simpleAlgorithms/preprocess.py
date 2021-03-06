'''
@author Theodore Morley
'''

from nltk.corpus import stopwords
from random import shuffle
import string
import sys
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
    negUniqueWords = set(strNeg.split(' '))
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
        x[count] /= max(1, np.sum(x[count]))
        y[count] = 1
        count+=1
        negLine = nEx.split(' ')
        for token in negLine:
            if token in w2i:
                x[count][w2i[token]] += 1
            else:
                pass
        x[count] /= max(1, np.sum(x[count]))
        y[count] = 0
        count += 1
    return x, y

def translateExamplesRT(w2i, trainingPos, trainingNeg, testPos, testNeg):
    # Initialize numpy arrays for training and test
    training_x = np.zeros((len(trainingPos)+len(trainingNeg), len(w2i)))
    training_y = np.zeros(len(trainingPos)+len(trainingNeg))
    test_x = np.zeros((len(testPos)+len(testNeg), len(w2i)))
    test_y = np.zeros(len(testPos)+len(testNeg))
    # Fill out the training arrays with frequencies
    training_x, training_y = countFreqs(w2i, trainingPos, trainingNeg, training_x, training_y)
    test_x, test_y = countFreqs(w2i, testPos, testNeg, test_x, test_y)
    return training_x, training_y, test_x, test_y

def termFrequencyNormalize(data):
    for row in range(len(data)):
        for col in range(len(data[row])):
            data[row][col] = data[row][col]/np.sum(data[row])
    return data

def fillSeqs(w2i, pos, neg, y):
    count = 0
    x = []
    for pEx, nEx in zip(pos, neg):
        posLine = pEx.split(' ')
        newLine = []
        for i in range(len(posLine)):
            if posLine[i] in w2i:
                # Adding one reserves 0 as the UNK token
                newLine.append(w2i[posLine[i]]+1)
            else:
                pass
        x.append(newLine)
        y[count] = 1
        count += 1
        newLine = []
        negLine = nEx.split(' ')
        for i in range(len(negLine)):
            if negLine[i] in w2i:
                newLine.append(w2i[negLine[i]]+1)
            else:
                pass
        x.append(newLine)
        y[count] = 0
        count += 1
    return np.array(x), y


def sequenceTranslation(w2i, trainingPos, trainingNeg, testPos, testNeg):
    training_x = np.zeros((len(trainingPos)+len(trainingNeg)))
    training_y = np.zeros((len(trainingPos)+len(trainingNeg)))
    test_x = np.zeros((len(testPos)+len(testNeg)))
    test_y = np.zeros((len(testPos)+len(testNeg)))
    training_x, training_y = fillSeqs(w2i, trainingPos, trainingNeg, training_y)
    test_x, test_y = fillSeqs(w2i, testPos, testNeg, test_y)
    return training_x, training_y, test_x, test_y

def getMaxLen(examples, sofar):
    for example in examples:
        l = len(example.split(' '))
        if l>sofar:
            sofar = l
    return sofar

if __name__ == '__main__':
    print('Enter \'rt\' if you are processing the RT dataset, or \'imdb\' if you are processing the imdb dataset.')
    dataset = sys.stdin.readline()
    dataset = dataset.strip()
    print('Please input the names/paths of the positive and negative files, space separated: ')
    paths = sys.stdin.readline()
    paths = paths.strip()
    paths = paths.split(' ')
    if dataset=='rt':
        # Get the train-test split
        posTrain, posTest = trainTestSplit(paths[0], 533)
        negTrain, negTest = trainTestSplit(paths[1], 533)
        # Get the unique vocab from training
        uniques = getVocab(posTrain, negTrain)
        # Get the w2ix
        w2x = word2ix(uniques, True, False)
        print('If you wish to translate to a sequence for LSTM input, please input \'lstm\', otherwise input \'standard\':')
        mode = sys.stdin.readline()
        mode = mode.strip()
        if mode=='standard':
            # Translate the examples
            tr_x, tr_y, ts_x, ts_y = translateExamplesRT(w2x, posTrain, negTrain, posTest, negTest)
            #tr_x = termFrequencyNormalize(tr_x)
            #ts_x = termFrequencyNormalize(ts_x)
            #Save the arrays
            np.save('training_x_RT', tr_x)
            np.save('training_y_RT', tr_y)
            np.save('test_x_RT', ts_x)
            np.save('test_y_RT', ts_y)
        elif mode == 'lstm':
            maxsofar= getMaxLen(posTrain+negTrain+negTest+posTest, 1)
            tr_x, tr_y, ts_x, ts_y = sequenceTranslation(w2x, posTrain, negTrain, posTest, negTest)
            np.save('training_x_RT_seq', tr_x)
            np.save('training_y_RT_seq', tr_y)
            np.save('test_x_RT_seq', ts_x)
            np.save('test_y_RT_seq', ts_y)
        print('Translation complete.')
    elif dataset=='imdb':
        pass
