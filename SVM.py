import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


def Vectorize(fileName, vectorizer):
    data = (open(fileName, 'rb').read().splitlines())
    return vectorizer.fit_transform(data).toarray()

def genVectorizer():
    wordList = pickle.load(open('./SVMData/WordList.txt', 'rb'))    
    vectorizer = CountVectorizer(vocabulary= wordList)
    vectorizer._validate_vocabulary()
    return vectorizer

def SVM():
    vectorizer = genVectorizer()
    negTrain = Vectorize('./SVMData/train_neg.txt', vectorizer)
    posTrain = Vectorize('./SVMData/train_pos.txt', vectorizer)
    Train = np.concatenate((negTrain, posTrain), 0)

    negY = np.full((negTrain.shape[0]), 0)
    posY = np.full((posTrain.shape[0]), 1)
    Y = np.concatenate((negY, posY), 0 )

    Train , Y = shuffleInputs(Train, Y)
    clf = trainSVM(Train, Y)
    return accuracy(clf, vectorizer)

def accuracy(clf, vectorizer):
    count = 0
    negTest = Vectorize('./SVMData/test_neg.txt', vectorizer)
    for i in testSVM(clf, negTest):
        if i == 0:
            count += 1
    posTest = Vectorize('./SVMData/test_pos.txt', vectorizer)
    for i in testSVM(clf, posTest):
        if i == 1:
            count += 1
    return float(count) / float(negTest.shape[0] + posTest.shape[0])


def shuffleInputs(a,b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    return  c[:, :a.size//len(a)].reshape(a.shape) , c[:, a.size//len(a):].reshape(b.shape)

def trainSVM(x,y):
    clf = svm.SVC()
    return clf.fit(x,y)
    
def testSVM(clf, x):
    return clf.predict(x)

if __name__== '__main__':
    SVM()
