import os, string, re, operator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

def union(a,b):
    return list(set(a)|set(b))

def unionList(data):
    output = []
    for i in data:
        output = union(output, i)
    with open('./SVMData/WordList.txt', 'wb') as f:
        pickle.dump(output, f)
    return output
    

def SVMDataProcess():
    dataList = []
    for root, dirs, files in os.walk('./data'):
        for file in files:
            print(file)
            words = {}
            if file.endswith(".txt"):
                name = './data/' + file
                output = './SVMData/' + file
                with open(name, 'r+ ') as f, open(output,'w') as out:
                    data = f.read().splitlines()
                    for line in data:
                        wordList = removeStopWords(re.sub("[^\w]", " ",  line.lower()))
                        out.write(" ".join(str(x) for x in wordList) + '\n')
                        for i in wordList:
                            if i not in words:
                                words[i] = 1
                            else:
                                words[i] +=1
                    dataList.append(list(words.keys()))
    return dataList

def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence

if __name__ == '__main__':  
    dataList = SVMDataProcess()
    unionList(dataList)


    