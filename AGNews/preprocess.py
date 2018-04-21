import csv
from nltk.corpus import stopwords
import string
import re

# path to AGNews raw dataset
raw_train = './AGNews_data/train.csv'
raw_test = './AGNews_data/test.csv'

# output path 
proc_train = './AGNews_data/proc_train.txt'
proc_test = './AGNews_data/proc_test.txt'

stops = set(stopwords.words('english'))
punct = string.punctuation

# def normalize(text):
#     text = re.sub(r"\\", " ", text)
#     text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
#     text = re.sub(r"\'s", " \'s", text)
#     text = re.sub(r"\'ve", " \'ve", text)
#     text = re.sub(r"n\'t", " n\'t", text)
#     text = re.sub(r"\'re", " \'re", text)
#     text = re.sub(r"\'d", " \'d", text)
#     text = re.sub(r"\'ll", " \'ll", text)
#     text = re.sub(r",", " , ", text)
#     text = re.sub(r"!", " ! ", text)
#     text = re.sub(r"\(", " ( ", text)
#     text = re.sub(r"\)", " ) ", text)
#     text = re.sub(r"\?", " ? ", text)
#     text = re.sub(r"\s{2,}", " ", text)
#     return text.strip().lower()

def clean(text):
    # make everything lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans('', '', punct))

    # remove stopwords
    words = text.split(" ")
    filtered_words = [word for word in words if word not in stops]

    return " ".join(filtered_words)


def preprocess(raw_path, output_path):
    with open(raw_path, 'r') as raw_file:
        with open(output_path, 'w') as proc_file:
            reader = csv.reader(raw_file, delimiter=',', quotechar='"')

            for row in reader:
                attr = clean(row[1] + " " + row[2])
                label = row[0]
                proc_file.write(attr + ',' + label + '\n')

if __name__ == '__main__':
    preprocess(raw_train, proc_train)
    preprocess(raw_test, proc_test)