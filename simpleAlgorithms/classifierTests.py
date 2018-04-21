'''
@author: Theodore Morley
'''
from sklearn import naive_bayes
from sklearn import svm
import numpy as np


def testClf(train_x, train_y, test_x, test_y, clf):
    clf.fit(train_x, train_y)
    return clf.score(test_x, test_y)

if __name__=='__main__':
    training_x = np.load('training_x_RT.npy')
    training_y = np.load('training_y_RT.npy')
    test_x = np.load('test_x_RT.npy')
    test_y = np.load('test_y_RT.npy')
    print("Running Naive Bayes...")
    nbResult = testClf(training_x, training_y, test_x, test_y, naive_bayes.GaussianNB())
    print("Naive Bayes Score: ")
    print(nbResult)
    print("Running SVM...")
    svmResult = testClf(training_x, training_y, test_x, test_y, svm.LinearSVC())
    print("SVM Score: ")
    print(svmResult)
