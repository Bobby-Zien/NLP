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

        self.training_file = "../feature_files/training_vec.feature"
        self.training_vec_npzfile="../pkg/training_vec.npz"
        self.dev_file = "../feature_files/dev_vec.feature"
        self.test_file = "../feature_files/test_vec.feature"


    def generate_file(self, file_type):
        stemmer = PorterStemmer()

        if file_type == "train":
            w = open(self.training_file, "w")
            r = open(self.training_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
            with open('../pkg/vecDict.pkl', 'rb') as vd:
                vecDict = pickle.load(vd)
        elif file_type == "dev":
            w = open(self.dev_file, "w")
            r = open(self.development_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
            with open('../pkg/vecDict_dev.pkl', 'rb') as vd:
                vecDict = pickle.load(vd)
        elif file_type == "test":
            w = open(self.test_file, "w")
            r = open(self.test_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
            with open('../pkg/vecDict_test.pkl', 'rb') as vd:
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
        token_tops=[]
        token_top_before=[]
        token_tops_before = []
        token_top2_before = []
        token_top_next = []
        token_tops_next=[]
        token_top_next2 = []
        target=[]

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
                target.append('\n')
                if token_top:
                    sorted_token_5=sorted(token_top,reverse=True)[:5]
                    sorted_token_3=sorted_token_5[:3]
                    sorted_token_1=sorted_token_3[0]
                    for t in token_top:
                        if t in sorted_token_5:
                            if t in sorted_token_3:
                                if t==sorted_token_1:
                                    token_tops += ['\ttop5=1\ttop3=1\ttop1=1\t']
                                else:
                                    token_tops += ['\ttop5=1\ttop3=1\ttop1=0\t']
                            else:
                                token_tops += ['\ttop5=1\ttop3=0\ttop1=0\t']
                        else:
                            token_tops+=['\ttop5=0\ttop3=0\ttop1=0\t']
                    token_tops.append('\n')
                    token_top=[]


                if token_top_next:
                    sorted_token_next_5 = sorted(token_top_next, reverse=True)[:5]
                    sorted_token_next_3 = sorted_token_next_5[:3]
                    sorted_token_next_1 = sorted_token_next_3[0]
                    for t in token_top_next:
                        if t in sorted_token_next_5:
                            if t in sorted_token_next_3:
                                if t == sorted_token_next_1:
                                    token_tops_next += ['\ttop5_next=1\ttop3_next=1\ttop1_next=1\t']
                                else:
                                    token_tops_next += ['\ttop5_next=1\ttop3_next=1\ttop1_next=0\t']
                            else:
                                token_tops_next += ['\ttop5_next=1\ttop3_next=0\ttop1_next=0\t']
                        else:
                            token_tops_next += ['\ttop5_next=0\ttop3_next=0\ttop1_next=0\t']
                    token_tops_next.append('\n')
                    token_top_next = []

                if token_top_before:
                    sorted_token_before_5 = sorted(token_top_before, reverse=True)[:5]
                    sorted_token_before_3 = sorted_token_before_5[:3]
                    sorted_token_before_1 = sorted_token_before_3[0]
                    for t in token_top_before:
                        if t in sorted_token_before_5:
                            if t in sorted_token_before_3:
                                if t == sorted_token_before_1:
                                    token_tops_before += ['\ttop5_before=1\ttop3_before=1\ttop1_before=1\t']
                                else:
                                    token_tops_before += ['\ttop5_before=1\ttop3_before=1\ttop1_before=0\t']
                            else:
                                token_tops_before += ['\ttop5_before=1\ttop3_before=0\ttop1_before=0\t']
                        else:
                            token_tops_before += ['\ttop5_before=0\ttop3_before=0\ttop1_before=0\t']
                    token_tops_before.append('\n')
                    token_top_before = []

                continue

            line = line.split()
            token = line[0]
            POS = line[1]
            BIO = line[2]
            tokenId = line[3]
            stem = stemmer.stem(token)

            # token_vec = vecDict[token]
            # try:
            #     token_sim = cosine_similarity(vec_avg, token_vec)
            # except:
            #     token_sim = 1


            if BIO[-2:]=='NP':
                token_vec=vecDict[token]
                try:
                    token_sim=cosine_similarity(vec_avg,token_vec)
                except:
                    token_sim =1
            else:
                token_sim=0


            token_top.append([token_sim,tokenId])

            ARG = "NONE"
            if len(line) > 5:
                ARG = line[5]

            # initialize prev and next variables
            prev_word = "BEGIN"
            prev_word2 = "BEGIN2"  # prev 2 word back

            next_word = "NEXT"
            next_word2 = "NEXT2"  # next 2 word forward

            # initialize prev and next variables with 0
            # prev_word_sim=0
            # next_word_sim=0

            if BIO[-2:] == 'NP':
                # check if prev_word and next_word exists
                if i > 0 and i < len(lines) - 1:
                    prev_line = lines[i - 1].split()
                    next_line = lines[i + 1].split()

                    if prev_line:
                        prev_word = prev_line[0]
                        prev_word_vec=nlp(prev_word+' '+token).vector
                        try:
                            prev_word_sim=cosine_similarity(prev_words_avg,prev_word_vec)
                        except:
                            prev_word_sim=1


                    if next_line:
                        next_word = next_line[0]
                        next_word_vec = nlp(token+' '+next_word).vector
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
                        prev_word2_vec=nlp(prev_word2+' '+prev_word+' '+token).vector
                        try:
                            prev_word2_sim= cosine_similarity(prev_word2s_avg, prev_word2_vec)
                        except:
                            prev_word2_sim=1

                    if next_line2:
                        next_word2 = next_line2[0]
                        next_word2_vec = nlp(token+' '+next_word+' '+next_word2).vector
                        try:
                            next_word2_sim= cosine_similarity(next_word2s_avg, next_word2_vec)
                        except:
                            next_word2_sim=1

            else:
                prev_word_sim=0
                next_word2_sim=0
                next_word_sim=0
                prev_word2_sim=0

            # l = "{}\tPOS={}\tstem={}\tBIO={}\tends={}\ttoken_sim={}".format(token, POS, stem, BIO, ends,token_sim)
            l = "{}\tPOS={}\tstem={}\tBIO={}\ttoken_sim={}".format(token, POS, stem, BIO, token_sim)


            token_top_before.append(prev_word_sim)
            token_top_next.append(next_word_sim)

            # if BIO[-2:]!='NP':
            #     prev_word_sim,next_word_sim,prev_word2_sim,next_word2_sim=0,0,0,0

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
                target.append("\t{}".format(ARG))


            ls.append(l)
            prev_word = token

        for i in range(len(ls)):
            if ls[i]!='\n':
                if file_type == "train":
                    w.write(ls[i]+token_tops[i]+token_tops_next[i]+token_tops_before[i]+target[i]+'\n')
                else:
                    w.write(ls[i] + token_tops[i]+token_tops_next[i]+token_tops_before[i] + '\n')
            else:
                w.write(ls[i])

        w.close()
        print(file_type + " completed")

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
                        prev_words.append(nlp(prev_word+' '+token).vector)

                    if next_line:
                        next_word = next_line[0]
                        next_words.append(nlp(token+' '+next_word).vector)

                # check if prev_word2 and next_word2 exists
                if i > 1 and i < len(lines) - 2:
                    prev_line2 = lines[i - 2].split()
                    next_line2 = lines[i + 2].split()

                    if prev_line2:
                        prev_word2 = prev_line2[0]
                        prev_word2s.append(nlp(prev_word2+' '+prev_word+' '+token).vector)

                    if next_line2:
                        next_word2 = next_line2[0]
                        next_word2s.append(nlp(token+' '+next_word+' '+next_word2).vector)


        vec_avg=average(array(vecs),axis=0)
        prev_words_avg = average(array(prev_words),axis=0)
        prev_word2s_avg = average(array(prev_word2s),axis=0)
        next_words_avg = average(array(next_words),axis=0)
        next_word2s_avg = average(array(next_word2s),axis=0)

        np.savez('training_vec_v1.npz',vec_avg=vec_avg,prev_words_avg=prev_words_avg,prev_word2s_avg=prev_word2s_avg,next_words_avg=next_words_avg,next_word2s_avg=next_word2s_avg)
        print("vector training completed!")

def main():
    input_file = "../input_files/part-training"
    dev_file = "../input_files/part-dev"
    test_file = "../input_files/part-test"

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