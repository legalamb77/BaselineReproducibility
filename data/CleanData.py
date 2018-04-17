import os

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith(".txt"):
            with open(file, 'r+ ') as f:
                data = f.read().splitlines()
                for line in range(len(data)):
                    data[line] = data[line].replace('<br />','')
                f.seek(0)
                for i in data:
                    f.write(i + '\n')
                f.truncate()