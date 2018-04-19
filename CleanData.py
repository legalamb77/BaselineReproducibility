import os

def cleanData():
    for root, dirs, files in os.walk('./data'):
        for file in files:
            if file.endswith(".txt"):
                with open('./data/'+file, 'r+ ') as f:
                    data = f.read().splitlines()
                    for line in range(len(data)):
                        data[line] = data[line].replace('<br />','').lower()
                    f.seek(0)
                    for i in data:
                        f.write(i + '\n')
                    f.truncate()

if __name__ == '__main__':  
    cleanData()
