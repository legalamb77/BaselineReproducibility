import csv
import re

raw_train = './AGNews_data/train.csv'
raw_test = './AGNews_data/test.csv'
proc_train = './AGNews_data/proc_train.txt'
proc_test = './AGNews_data/proc_test.txt'

def clean(raw_path, output_path):
    with open(raw_path, 'r') as raw_f:
        with open(output_path, 'w') as proc_f:
            reader = csv.reader(raw_f, delimiter=',', quotechar='"')

            for row in reader:
                txt = ""

                for col in row[1:]:
                    txt = txt + " " + col.replace("\\", " ")

                proc_f.write(str(int(row[0]) - 1) + '\t' + txt + '\n')

def preprocess():
    clean(raw_train, proc_train)
    clean(raw_test, proc_test)
    
preprocess()