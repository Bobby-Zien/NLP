"""
Group 4 - Maximizing the Score for Detecting ARG1 on Partitives

Zien Yang, zy2236, N17995064
Jingwei Ye, jy3555, N10236604
Zeyu Chen, zc2078, N10456612
"""

import sys
import pickle
from nltk.stem import *
import numpy as np
from numpy import array, average
from scipy import spatial
import spacy
nlp=spacy.load('en_core_web_md')

class Feature:
    def __init__(self, training_corpus, development_corpus, test_corpus):
        self.training_corpus = training_corpus
        self.development_corpus = development_corpus
        self.test_corpus = test_corpus

        self.training_file = "training_vec.feature"
        self.training_vec_npzfile="training_vec.npz"
        self.dev_file = "dev_vec.feature"
        self.test_file = "test_vec.feature"

    def generate_file(self, file_type):
        stemmer = PorterStemmer()

        if file_type == "train":
            w = open(self.training_file, "w")
            r = open(self.training_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
            with open('vecDict.pkl', 'rb') as vd:
                vecDict = pickle.load(vd)
        elif file_type == "dev":
            w = open(self.dev_file, "w")
            r = open(self.development_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
            with open('vecDict_dev.pkl', 'rb') as vd:
                vecDict = pickle.load(vd)
        elif file_type == "test":
            w = open(self.test_file, "w")
            r = open(self.test_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
            with open('vecDict_test.pkl', 'rb') as vd:
                vecDict = pickle.load(vd)

        data = np.load(self.training_vec_npzfile)
        vec_avg=data['vec_avg']
        prev_words_avg = data['prev_words_avg']
        prev_word2s_avg = data['prev_word2s_avg']
        next_words_avg = data['next_words_avg']
        next_word2s_avg = data['next_word2s_avg']

        cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

        n = len(lines)
        count = 0
        ls = []
        token_top=[]
        token_top_before=[]
        token_top2_before = []
        token_top_next = []
        token_top_next2 = []

        for i, line in enumerate(lines):

            count += 1
            if count == int(n / 4):
                print(file_type+":")
                print("25% complete")
            elif count == int(n / 2):
                print("50% complete")
            elif count == int(n / 4 * 3):
                print("75% complete")


            if line == "":
                ls.append("\n")
                if token_top:
                    sorted(token_top,reverse=True)[:5]
                continue

            line = line.split()
            token = line[0]
            token_vec=vecDict[token]
            try:
                token_sim=cosine_similarity(vec_avg,token_vec)
            except:
                token_sim =1


            POS = line[1]
            BIO = line[2]
            tokenId=line[3]
            stem = stemmer.stem(token)
            ends = token[-1]

            token_top=[]

            ARG = "NONE"
            if len(line) > 5:
                ARG = line[5]

            # initialize prev and next variables
            prev_word = "BEGIN"
            prev_word2 = "BEGIN2"  # prev 2 word back

            next_word = "NEXT"
            next_word2 = "NEXT2"  # next 2 word forward

            # check if prev_word and next_word exists
            if i > 0 and i < len(lines) - 1:
                prev_line = lines[i - 1].split()
                next_line = lines[i + 1].split()

                if prev_line:
                    prev_word = prev_line[0]
                    prev_word_vec=vecDict[prev_word]
                    try:
                        prev_word_sim=cosine_similarity(prev_words_avg,prev_word_vec)
                    except:
                        prev_word_sim=1

                if next_line:
                    next_word = next_line[0]
                    next_word_vec = vecDict[next_word]
                    try:
                        next_word_sim = cosine_similarity(next_words_avg, next_word_vec)
                    except:
                        next_word_sim=1

            # check if prev_word2 and next_word2 exists
            if i > 1 and i < len(lines) - 2:
                prev_line2 = lines[i - 2].split()
                next_line2 = lines[i + 2].split()

                if prev_line2:
                    prev_word2 = prev_line2[0]
                    prev_word2_vec=vecDict[prev_word2]
                    try:
                        prev_word2_sim= cosine_similarity(prev_word2s_avg, prev_word2_vec)
                    except:
                        prev_word2_sim=1

                if next_line2:
                    next_word2 = next_line2[0]
                    next_word2_vec = vecDict[next_word2]
                    try:
                        next_word2_sim= cosine_similarity(next_word2s_avg, next_word2_vec)
                    except:
                        next_word2_sim=1

            # l = "{}\tPOS={}\tstem={}\tBIO={}\tends={}\ttoken_sim={}".format(token, POS, stem, BIO, ends,token_sim)
            l = "{}\tPOS={}\tstem={}\tBIO={}\ttoken_sim={}".format(token, POS, stem, BIO, token_sim)

            if prev_word != "BEGIN":
                l += "\tprevious_word_sim={}\t".format(prev_word_sim)

            if next_word != "NEXT":
                l += "\tnext_word_sim={}".format( next_word_sim)

            if prev_word2 != "BEGIN2":
                l += "\tprevious_word2_sim={}".format(prev_word2_sim)

            if next_word2 != "NEXT2":
                l += "\tnext_word2_sim={}".format( next_word2_sim)

            # add ARG to the training file
            if file_type == "train":
                l += "\t{}\n".format(ARG)

            # do not add ARG to the dev and test
            else:
                l += "\n"

            ls.append(l)
            prev_word = token

        for lw in ls:
            w.write(lw)

        w.close()
        print(file_type + " completed")

    def generateVecDict(self,file_type):

        if file_type == "train":
            r = open(self.training_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
        elif file_type == "dev":
            r = open(self.development_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
        elif file_type == "test":
            r = open(self.test_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()

        n = len(lines)
        count = 0
        vecDict={}
        for i, line in enumerate(lines):
            count += 1
            if count == int(n / 4):
                print("25% complete")
            elif count == int(n / 2):
                print("50% complete")
            elif count == int(n / 4 * 3):
                print("75% complete")

            if line == "":
                continue

            line = line.split()
            token = line[0]
            if token not in vecDict:
                token_vec=nlp(token).vector
                vecDict[token]=token_vec

        outputName='vecDict_'+file_type+'.pkl'
        with open(outputName, 'wb') as f:
            pickle.dump(vecDict, f)



    def trainvector(self):
        r = open(self.training_corpus, 'r', encoding='utf8')
        lines = r.read().splitlines()
        r.close()

        vecs=[]
        prev_words = []
        prev_word2s = []
        next_words = []
        next_word2s = []
        n=len(lines)
        count=0
        for i, line in enumerate(lines):
            count+=1
            if count==int(n/4):
                print("Training average vector:")
                print("25% complete")
            elif count==int(n/2):
                print("50% complete")
            elif count==int(n/4*3):
                print("75% complete")


            if line == "":
                continue

            line = line.split()
            token = line[0]

            ARG = "NONE"
            if len(line) > 5:
                ARG = line[5]


            if ARG=="ARG1":
                vec = nlp(token).vector
                vecs.append(vec)

                # initialize prev and next variables
                prev_word = "BEGIN"
                prev_word2 = "BEGIN2"  # prev 2 word back


                next_word = "NEXT"
                next_word2 = "NEXT2"  # next 2 word forward


                # check if prev_word and next_word exists
                if i > 0 and i < len(lines) - 1:
                    prev_line = lines[i - 1].split()
                    next_line = lines[i + 1].split()

                    if prev_line:
                        prev_word = prev_line[0]
                        prev_words.append(nlp(prev_word).vector)

                    if next_line:
                        next_word = next_line[0]
                        next_words.append(nlp(next_word).vector)

                # check if prev_word2 and next_word2 exists
                if i > 1 and i < len(lines) - 2:
                    prev_line2 = lines[i - 2].split()
                    next_line2 = lines[i + 2].split()

                    if prev_line2:
                        prev_word2 = prev_line2[0]
                        prev_word2s.append(nlp(prev_word2).vector)

                    if next_line2:
                        next_word2 = next_line2[0]
                        next_word2s.append(nlp(next_word2).vector)


        vec_avg=average(array(vecs),axis=0)
        prev_words_avg = average(array(prev_words),axis=0)
        prev_word2s_avg = average(array(prev_word2s),axis=0)
        next_words_avg = average(array(next_words),axis=0)
        next_word2s_avg = average(array(next_word2s),axis=0)

        np.savez('training_vec.npz',vec_avg=vec_avg,prev_words_avg=prev_words_avg,prev_word2s_avg=prev_word2s_avg,next_words_avg=next_words_avg,next_word2s_avg=next_word2s_avg)
        print("vector training completed!")


def main():
    input_file = "input_files/part-training"
    dev_file = "input_files/part-dev"
    test_file = "input_files/part-test"

    if len(sys.argv) > 3:
        input_file = sys.argv[1]
        dev_file = sys.argv[2]
        test_file = sys.argv[3]

    feat = Feature(input_file, dev_file, test_file)
    # feat.trainvector()
    # feat.generateVecDict(file_type="train")
    feat.generate_file(file_type="train")
    # feat.generate_file(file_type="dev")
    # feat.generateVecDict(file_type="test")
    feat.generate_file(file_type="test")


if __name__ == "__main__":
    main()