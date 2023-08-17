from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.tokenize import sent_tokenize
import numpy as np
import sys, string

##
#Usage: python novelty.py Gold_File.txt Generation_File.txt
##
# lemmatizer = WordNetLemmatizer()
def normalization(text):
    a = text.split()
    new_list = []
    for e in a:
        e = lemmatizer.lemmatize(e)
        new_list.append(e)
    return new_list

def jaccard_similarity(text1, text2):
    a = text1.split() #normalization(text1)
    b = text2.split() #normalization(text2)

    intersection = len(list(set(a).intersection(set(b))))   #len(list(set(a).intersection(set(b))))
    union = len(set(a).union(set(b)))    #(len(a) + len(b)) - intersection
    if union == 0:
        print(a, b)
    return float(intersection) / float(union)

def novelty(train, generation):
    novelty_score = []
    for t in generation:
        t_score = []
        for text in train:
            t_score.append(jaccard_similarity(text, t))
        s = 1 - max(t_score)
        novelty_score.append(s)
    return np.mean(np.array(novelty_score))

def read_txt(filename):
    sentences = []
    with open(filename) as f:
        for line in f:
            h = line.translate(str.maketrans('', '', string.punctuation))
            h = " ".join(h.lower().split())
            sentences.append(h)
    return sentences

if __name__ == "__main__":

    gold = read_txt(sys.argv[1])   	    # ".txt"
    generation = read_txt(sys.argv[2])      # ".txt"

    score = novelty(list(set(gold)), generation)
    print("novelty for {} is {:.4f}.".format(sys.argv[2], score))