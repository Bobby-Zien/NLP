"""
Zien Yang
zy2236
N17995064
"""

import sys
from nltk.stem import *

class HW6:
    def __init__(self, training_corpus : str, development_corpus: str) -> None:
        self.training_corpus = training_corpus
        self.development_corpus = development_corpus

        self.training_file = "training.feature"
        self.test_file = "test.feature"

    def generate_file(self, file_type : str):
        stemmer = PorterStemmer()

        if file_type == "train":
            w = open(self.training_file, "w")
            r = open(self.training_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()
        elif file_type == "test":
            w = open(self.test_file, "w")
            r = open(self.development_corpus, 'r', encoding='utf8')
            lines = r.read().splitlines()
            r.close()

        for i, line in enumerate(lines):
            if line == "":
                w.write("\n")
                continue

            line = line.split()
            token = line[0]
            POS = line[1]
            BIO = line[2]
            stem = stemmer.stem(token)
            ends = token[-1]
            
            ARG = "NONE"
            if len(line) > 5:
                ARG = line[5]
            
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
                    
            l = "{}\tPOS={}\tstem={}\tBIO={}\tends={}".format(token, POS, stem, BIO, ends)

            if prev_POS != "BEGIN":
                l += "\tprevious_POS={}\tprevious_word={}\t".format(prev_POS, prev_word)

            if next_POS != "NEXT":
                l += "\tnext_POS={}\tnext_word={}".format(next_POS, next_word)

            if prev_POS2 != "BEGIN2":
                l += "\tprevious_POS2={}\tprevious_word2={}".format(prev_POS2, prev_word2)

            if next_POS2 != "NEXT2":
                l += "\tnext_POS2={}\tnext_word2={}".format(next_POS2, next_word2)

            # add ARG to the training file
            if file_type != "test":
                l += "\t{}\n".format(ARG)
            
            # do not add BIO to the dev and test
            else:
                l += "\n"
            
            w.write(l)
            prev_word = token
            prev_POS = POS

        w.close()

def main():
    input_file = "part-training"
    test_file = "part-dev"

    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        test_file = sys.argv[2]

    hw6 = HW6(input_file, test_file)
    hw6.generate_file(file_type="train")
    hw6.generate_file(file_type="test")

if __name__ == "__main__":
    main()
