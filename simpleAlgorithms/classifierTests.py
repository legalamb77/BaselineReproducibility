'''
@author: Theodore Morley
'''
from sklearn import naive_bayes
from sklearn import svm
import numpy as np
import time


def testClf(train_x, train_y, test_x, test_y, clf):
    clf.fit(train_x, train_y)
    return clf.score(test_x, test_y)

if __name__=='__main__':
    training_x = np.load('training_x_RT.npy')
    training_y = np.load('training_y_RT.npy')
    test_x = np.load('test_x_RT.npy')
    test_y = np.load('test_y_RT.npy')
    print("Running Naive Bayes...")
    t0 = time.time()
    nbResult = testClf(training_x, training_y, test_x, test_y, naive_bayes.MultinomialNB())
    print("Naive Bayes Score: ")
    print(nbResult)
    t1 = time.time()
    totalNB = t1-t0
    print("Running SVM...")
    t2 = time.time()
    svmResult = testClf(training_x, training_y, test_x, test_y, svm.LinearSVC())
    t3 = time.time()
    totalSVM = t3-t2
    print("SVM Score: ")
    print(svmResult)
    print("NB Time: ")
    print(totalNB)
    print("SVM Time: ")
    print(totalSVM)
