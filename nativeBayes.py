import sys
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np

training_features = []
training_labels = []
testing_features = []

training_lines = []
testing_lines = []

word_map = dict()
word_count = 1
BIO_map = dict()
BIO_count = 1
pos_map = dict()
pos_count = 1
label_map = dict()
label_count = 1
label_reverse_map = dict()
# the longest sentence length in the training file
max_sentence_len = 0
# model
nb = MultinomialNB()

# nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#               ])


def extractFile(file_name):
    file = open(file_name)
    lines = file.readlines()
    lines = [line.split() for line in lines]
    return lines


def constructMap(train_file_name, test_file_name):
    global training_lines
    global testing_lines
    global label_count
    global word_count
    global BIO_count
    global pos_count
    training_lines = extractFile(train_file_name)
    testing_lines = extractFile(test_file_name)
    print(f'the length of the training lines is {len(training_lines)}')
    print(f'the length of the testing lines is {len(testing_lines)}')
    total_lines = training_lines + testing_lines
    print(f'the length of the total lines is {len(total_lines)}')
    # print(total_lines[len(training_lines)])
    for i in range(len(total_lines)):
        if total_lines[i]:
            # labels only exist in training lines
            if i < len(training_lines):
                if total_lines[i][-1] not in label_map:
                    label_map[total_lines[i][-1]] = label_count
                    label_reverse_map[label_count] = total_lines[i][-1]
                    label_count += 1

            # word
            if total_lines[i][0] not in word_map:
                word_map[total_lines[i][0]] = word_count
                word_count += 1
            # POS, stem, BIO
            for element in total_lines[i]:
                if "POS=" in element:
                    pos = element.split("=")[1]
                    if pos not in pos_map:
                        pos_map[pos] = pos_count
                        pos_count += 1
                if "stem=" in element:
                    stem = element.split("=")[1]
                    if stem not in word_map:
                        word_map[stem] = word_count
                        word_count += 1
                if "BIO=" in element:
                    bio = element.split("=")[1]
                    if bio not in BIO_map:
                        BIO_map[bio] = BIO_count
                        BIO_count += 1

def encodeFile(file_type):
    if file_type == "training":
        global training_features
        global training_labels
        for line in training_lines:
            if line:
                curr_line = []
                for i in range(len(line)):
                    if i == 0:
                        curr_line.append(word_map[line[i]])
                    elif i == len(line) - 1:
                        training_labels.append(label_map[line[i]])
                    else:
                        if "POS=" in line[i]:
                            pos = line[i].split("=")[1]
                            curr_line.append(pos_map[pos])
                        elif "stem=" in line[i] or "word=" in line[i]:
                            word = line[i].split("=")[1]
                            curr_line.append(word_map[word])
                        elif "BIO=" in line[i]:
                            bio = line[i].split("=")[1]
                            curr_line.append(BIO_map[bio])
                training_features.append(curr_line)

        print(f'the length of training features is {len(training_features)}')
        print(f'the length of training labels is {len(training_labels)}')

    elif file_type == "testing":
        for line in testing_lines:
            if line:
                curr_line = []
                for i in range(len(line)):
                    if i == 0:
                        curr_line.append(word_map[line[i]])
                    elif "POS=" in line[i]:
                        pos = line[i].split("=")[1]
                        curr_line.append(pos_map[pos])
                    elif "stem=" in line[i] or "word=" in line[i]:
                        word = line[i].split("=")[1]
                        curr_line.append(word_map[word])
                    elif "BIO=" in line[i]:
                        bio = line[i].split("=")[1]
                        curr_line.append(BIO_map[bio])
            testing_features.append(curr_line)
        print(f'the length of testing features is {len(testing_features)}')
    else:
        sys.exit('invalid file type')


# read the training lines and split it to features and labels
# fit the model using features and labels
def fitModel():
    global max_sentence_len
    encodeFile("training")
    max_sentence_len = max(map(len, training_features))
    print(f'the longest sentence length in the training file is {max_sentence_len}')
    features = np.array([np.pad(sentence, (0, max_sentence_len - len(sentence)), 'constant') for sentence in training_features])
    print(f'the shape of training features is {features.shape}')
    labels = np.array(training_labels, dtype=object)
    print(f'the shape of training labels is {labels.shape}')
    labels = labels.astype('int')
    nb.fit(features, labels)

    # nb.fit(features, labels)


def predict():
    encodeFile("testing")
    features = np.array(
        [np.pad(sentence, (0, max_sentence_len - len(sentence)), 'constant') for sentence in testing_features])
    print(f'the shape of testing feature is {features.shape}')
    label_pred = nb.predict(features)
    print(f'the length of prediction output is {len(label_pred)}')
    write_output(testing_lines, label_pred)


def write_output(testing_lines, label_pred):
    out = open('nb_output.txt', 'w+')
    for i in range(len(testing_lines)):
        if not testing_lines[i]:
            out.write('\n')
        else:
            out.write(testing_lines[i][0] + '\t')
            out.write(label_reverse_map[label_pred[i]])
            out.write('\n')


def main(args):
    training_file = args[1]
    testing_file = args[2]
    constructMap(training_file, testing_file)
    fitModel()
    predict()


if __name__ == '__main__': sys.exit(main(sys.argv))
