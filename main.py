import sys
import random
import numpy as np

lang = "hi"
file = open(lang+".txt")
temp = file.read().split("\n--***--\n")
data = []
for i in range(len(temp)):
    data.append(temp[i].split("\n"))

phonemes = set(())
p_classes = {}
c_phonemes = {}
n_syllable = []
syllable_p = []
onsets = []
onset_p = []
nuclei = []
nucleus_p = []
codas = []
coda_p = []
s_rep_f = []
s_rep_r = []
s_rep_p = []
w_rep_f = []
w_rep_r = []
w_rep_p = []
for i in range(len(data[0])):
    temp = data[0][i].split(":")
    phonemes.add(temp[0])
    p_classes[temp[0]] = temp[1]
    for j in range(len(temp[1])):
        try:
            c_phonemes[temp[1][j]].add(temp[0])
        except KeyError:
            c_phonemes[temp[1][j]] = set((temp[0]))
for i in range(len(data[1])):
    temp = data[1][i].split(":")
    n_syllable.append(int(temp[0]))
    syllable_p.append(float(temp[1]))
for i in range(len(data[2])):
    temp = data[2][i].split(":")
    onsets.append(temp[0])
    onset_p.append(float(temp[1]))
for i in range(len(data[3])):
    temp = data[3][i].split(":")
    nuclei.append(temp[0])
    nucleus_p.append(float(temp[1]))
for i in range(len(data[4])):
    temp = data[4][i].split(":")
    codas.append(temp[0])
    coda_p.append(float(temp[1]))
for i in range(len(data[5])):
    temp = data[5][i].split(":")
    s_rep_f.append(temp[0])
    s_rep_r.append(temp[1])
    s_rep_p.append(float(temp[2]))
for i in range(len(data[6])):
    temp = data[6][i].split(":")
    w_rep_f.append(temp[0])
    w_rep_r.append(temp[1])
    w_rep_p.append(float(temp[2]))
classes = list(c_phonemes.keys())

def randomIn(list):
    return list[random.randint(0, len(list)-1)]

def fromClass(c):
    possible = phonemes
    for i in range(len(c)):
        possible = possible & c_phonemes[c[i]]
    return randomIn(list(possible))

def ifProb(p):
    return random.random() <= p

def choiceP(choices, prob):
    return np.random.choice(choices, p=np.asarray(prob)/np.sum(prob))
def makeSyllable():
    syllable = choiceP(onsets, onset_p)+"."
    syllable += choiceP(nuclei, nucleus_p)+"."
    syllable += choiceP(codas, coda_p)
    pieces = syllable.split(".")
    for i in range(len(pieces)):
        if not pieces[i] == '' and pieces[i][0] in classes:
            syllable = syllable.replace(pieces[i], fromClass(pieces[i]))
    syllable = syllable.replace(".", "")
    sys.stdout.write(syllable+ "/")
    syllable = "#"+syllable+"# "
    for i in range(len(s_rep_f)):
        if s_rep_p[i] == 1 or ifProb(s_rep_p[i]):
            syllable = syllable.replace(s_rep_f[i], s_rep_r[i])
    for i in range(len(s_rep_f)):
        if s_rep_p[i] == 1:
            syllable = syllable.replace(s_rep_f[i], s_rep_r[i])
    syllable = syllable[1:-2]
    sys.stdout.write(syllable+ " ")
    return syllable
def makeWord():
    num = choiceP(n_syllable, syllable_p)
    word = ""
    print("")
    for i in range(num):
        word += makeSyllable()
    word = "~#"+word+"# "
    for i in range(len(word)-2, -1, -1):
        if word[i] == word[i+1]:
            word = word[:i]+word[i+1:]
    sys.stdout.write(word[2:-2]+"/")
    for i in range(len(w_rep_f)):
        if w_rep_p[i] == 1 or ifProb(w_rep_p[i]):
            word = word.replace(w_rep_f[i], w_rep_r[i])
    for i in range(len(w_rep_f)):
        if w_rep_p[i] == 1:
            word = word.replace(w_rep_f[i], w_rep_r[i])
    word = word[2:-2]
    sys.stdout.write(word+" ")
    return word
def makeSentence(num):
    sentence = ""
    for i in range(num-1):
        sentence += makeWord() + " "
    sentence += makeWord()+"."
    print("\n")
    return sentence

print(makeSentence(5))

