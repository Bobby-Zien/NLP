"""
Group 4 - Maximizing the Score for Detecting ARG1 on Partitives

Zien Yang, zy2236, N17995064
Jingwei Ye, jy3555, N10236604
Zeyu Chen, zc2078, N10456612
"""

import sys
from nltk.stem import *

class Feature:
    def __init__(self, training_corpus : str, development_corpus : str, test_corpus : str) -> None:
        self.training_corpus = training_corpus
        self.development_corpus = development_corpus
        self.test_corpus = test_corpus

        self.training_file = "./feature_files/training_path.feature"
        self.dev_file = "./feature_files/dev_path.feature"
        self.test_file = "./feature_files/test_path.feature"

    def generate_file(self, file_type : str):
        stemmer = PorterStemmer()

        if file_type == "train":
            w = open(self.training_file, "w")
            r = open(self.training_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
        elif file_type == "dev":
            w = open(self.dev_file, "w")
            r = open(self.development_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
        elif file_type == "test":
            w = open(self.test_file, "w")
            r = open(self.test_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()

        path = []       # BIO tags from Pred to ARG1
        pred_idx = -1   # the index of the predicate of the current sentence

        for i, line in enumerate(lines):
            # new line
            if line == "":
                path = []
                pred_idx = -1
                w.write("\n")
                continue
            
            # find the predicate's index of the sentence
            if pred_idx == -1:
                j = i
                while j < len(lines):
                    if lines[j] == "":
                        break
                    next_line = lines[j].split()
                    if len(next_line) > 5 and next_line[5] == "PRED":
                        pred_idx = int(lines[j].split()[3])
                        break
                    j += 1

            line = line.split()
            token = line[0]
            POS = line[1]
            BIO = line[2]
            sentence_idx = int(line[3])
            stem = stemmer.stem(token)
            
            ARG = "NONE"
            if len(line) > 5:
                ARG = line[5]

            # store the BIO to the path if the word follows a PRED or it is a PRED
            if ARG == "PRED":
                path.append("PRED")
            elif (len(path) > 0 and BIO[0] == 'B'):
                path.append(BIO[2:])    # ignore first two chars "B-"
            
            # initialize prev and next variables
            prev_word = "BEGIN"
            prev_POS = "BEGIN"
            prev_word2 = "BEGIN2"   # prev 2 word back
            prev_POS2 = "BEGIN2"

            next_word = "NEXT"
            next_POS = "NEXT"
            next_word2 = "NEXT2"    # next 2 word forward
            next_POS2 = "NEXT2"
            
            # check if prev_word and next_word exists
            if i > 0 and i < len(lines) - 1:
                prev_line = lines[i-1].split()
                next_line = lines[i+1].split()

                if prev_line:
                    prev_word = prev_line[0]
                    prev_POS = prev_line[1]

                if next_line:
                    next_word = next_line[0]
                    next_POS = next_line[1]

            # check if prev_word2 and next_word2 exists
            if i > 1 and i < len(lines) - 2:
                prev_line2 = lines[i-2].split()
                next_line2 = lines[i+2].split()

                if prev_line2:
                    prev_word2 = prev_line2[0]
                    prev_POS2 = prev_line2[1]

                if next_line2:
                    next_word2 = next_line2[0]
                    next_POS2 = next_line2[1]
                    
            l = "{}\tPOS={}\tstem={}\tBIO={}".format(token, POS, stem, BIO)
            # l = f'{token}\t{POS}\t{stem}\t{BIO}\t{ends}'

            # path feature
            if (len(path)) > 0:
                l += "\tpath={}".format('_'.join(path))
            else:
                l += "\tpath=_"

            # distance from predicate
            l += "\tdistance={}".format(str(sentence_idx - pred_idx))

            if prev_POS != "BEGIN":
                l += "\tprevious_POS={}\tprevious_word={}".format(prev_POS, prev_word)
                # l += f'\t{prev_POS}\t{prev_word}'

            if next_POS != "NEXT":
                l += "\tnext_POS={}\tnext_word={}".format(next_POS, next_word)
                # l += f'\t{next_POS}\t{next_word}'

            if prev_POS2 != "BEGIN2":
                l += "\tprevious_POS2={}\tprevious_word2={}".format(prev_POS2, prev_word2)
                # l += f'\t{prev_POS2}\t{prev_word2}'
            if next_POS2 != "NEXT2":
                l += "\tnext_POS2={}\tnext_word2={}".format(next_POS2, next_word2)
                # l += f'\t{next_POS2}\t{next_word2}'
            # add ARG to the training file
            if file_type == "train":
                l += "\t{}\n".format(ARG)
            
            # do not add ARG to the dev and test
            else:
                l += "\n"
            
            w.write(l)
            prev_word = token
            prev_POS = POS

        w.close()

def main():
    input_file = "input_files/part-training"
    dev_file = "input_files/part-dev"
    test_file = "input_files/part-test"

    if len(sys.argv) > 3:
        input_file = sys.argv[1]
        dev_file = sys.argv[2]
        test_file = sys.argv[3]

    feat = Feature(input_file, dev_file, test_file)
    feat.generate_file(file_type="train")
    feat.generate_file(file_type="dev")
    feat.generate_file(file_type="test")

if __name__ == "__main__":
    main()
