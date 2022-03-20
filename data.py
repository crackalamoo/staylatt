#http://research.ics.aalto.fi/cog/data/udhr/
import numpy as np

WORD_LEN = 20

CODES_EN = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7,
    'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14,
    'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21,
    'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, ' ': 0}

def codeToStr(word, codes):
    inv = {v: k for k, v in codes.items()}
    s = ""
    for i in range(len(word)):
        s += inv[word[i]]
    return s

inp = open("data/eng.txt", "r")
corpus = inp.read().lower()
inp.close()
corpus = corpus.replace("\n", " ").replace("\t", " ")
for i in range(len(corpus)-1, -1, -1):
    if not corpus[i] in CODES_EN.keys():
        corpus = corpus[:i]+corpus[i+1:]
corpus = corpus.replace("  ", " ")
corpus = corpus.replace("  ", " ")
print(corpus)
corpus = corpus.split(" ")
data = []
for i in range(len(corpus)):
    temp = []
    for j in range(len(corpus[i])):
        temp.append(CODES_EN[corpus[i][j]])
    if len(temp) > 0:
        while len(temp) < WORD_LEN:
            temp.insert(0, 0)
            temp.append(0)
        while len(temp) > WORD_LEN:
            temp = temp[:-1]
        data.append(temp)
data = np.array(data).flatten()

np.savetxt("data/en.csv", data.astype(int), fmt="%i", delimiter=',')