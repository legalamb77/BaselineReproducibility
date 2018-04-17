import os, string, re, operator
for root, dirs, files in os.walk('.'):
    for file in files:
        words = {}
        if file.endswith(".txt"):
            name = './data/' + file
            output = './SVMData/' + file
            with open(name, 'r+ ') as f, open(output,'w') as out:
                data = f.read().splitlines()
                for line in data:
                    wordList = re.sub("[^\w]", " ",  line.lower()).split()
                    for i in wordList:
                        if i not in words:
                            words[i] = 1
                        else:
                            words[i] +=1
                data = sorted(words.items(), key = operator.itemgetter(1),reverse = True)
                for i in range(len(data)):
                    write = ''. join(str(data[i][0]) + "\t"+ "\t" + str(data[i][1]))
                    if(i != len(data)):
                        write += "\n"
                    out.write(write)