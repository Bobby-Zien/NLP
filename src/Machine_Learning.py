import sys
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


feature_selection = ""  # set by argv
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
stem_map = dict()
stem_count = 1
path_map = dict()
path_count = 1
# the longest sentence length in the training file
max_sentence_len = 0

# models
nb = MultinomialNB()
sgd = SGDClassifier()
decision_tree = tree.DecisionTreeClassifier()
forest = RandomForestClassifier()

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
    global stem_count
    global path_count
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
            # POS, stem, BIO, Path
            for element in total_lines[i]:
                if "POS=" in element in element:
                    pos = element.split("=")[1]
                    if pos not in pos_map:
                        pos_map[pos] = pos_count
                        pos_count += 1
                if "stem=" in element:
                    stem = element.split("=")[1]
                    if stem not in stem_map:
                        stem_map[stem] = stem_count
                        stem_count += 1
                if "BIO=" in element:
                    bio = element.split("=")[1]
                    if bio not in BIO_map:
                        BIO_map[bio] = BIO_count
                        BIO_count += 1
                if "path=" in element:
                    path = element.split("=")[1]
                    if path not in path_map:
                        path_map[path] = path_count
                        path_count += 1


def encodeFile(file_type):
    if file_type == "training":
        global training_features
        global training_labels
        global max_sentence_len
        training_features = []
        training_labels = []
        for line in training_lines:
            if line:
                curr_line = []
                for i in range(len(line)):
                    if i == 0:
                        curr_line.append(word_map[line[i]])
                    if i == len(line) - 1:
                        training_labels.append(label_map[line[i]])
                    else:
                        if "POS=" in line[i] or "POS2=" in line[i]:
                            pos = line[i].split("=")[1]
                            curr_line.append(pos_map[pos])
                        elif "stem=" in line[i]:
                            stem = line[i].split("=")[1]
                            curr_line.append(stem_map[stem])
                        elif "BIO=" in line[i]:
                            bio = line[i].split("=")[1]
                            curr_line.append(BIO_map[bio])
                        elif "sim=" in line[i]:
                            sim = (float(line[i].split("=")[1]) + 1)
                            if sim < 0:
                                sim = abs(sim)
                            else:
                                sim = sim * 10000
                            curr_line.append(str(sim))
                        elif "top5" in line[i] or "top3" in line[i] or "top1" in line[i]:
                            curr_line.append(line[i].split("=")[1])
                        elif "word=" in line[i] or "word2=" in line[i]:
                            word = line[i].split("=")[1]
                            curr_line.append(word_map[word])
                        elif "path=" in line[i]:
                            path = line[i].split("=")[1]
                            curr_line.append(path_map[path])
                        elif "distance=" in line[i]:
                            distance = int(line[i].split("=")[1]) + 200
                            curr_line.append(distance)

                training_features.append(curr_line)

        max_sentence_len = max(map(len, training_features))
        print(f'the longest sentence length in the training file is {max_sentence_len}')
        print(f'the length of training features is {len(training_features)}')
        print(f'the length of training labels is {len(training_labels)}')

    elif file_type == "testing":
        global testing_features
        testing_features = []
        for line in testing_lines:
            if line:
                curr_line = []
                for i in range(len(line)):
                    if i == 0:
                        curr_line.append(word_map[line[i]])
                    if "POS=" in line[i] or "POS2=" in line[i]:
                        pos = line[i].split("=")[1]
                        curr_line.append(pos_map[pos])
                    elif "stem=" in line[i]:
                        stem = line[i].split("=")[1]
                        curr_line.append(stem_map[stem])
                    elif "BIO=" in line[i]:
                        bio = line[i].split("=")[1]
                        curr_line.append(BIO_map[bio])
                    elif "sim=" in line[i]:
                        sim = (float(line[i].split("=")[1]) + 1)
                        if sim < 0:
                            sim = abs(sim)
                        else:
                            sim = sim * 10000
                        curr_line.append(str(sim))
                    elif "top5" in line[i] or "top3" in line[i] or "top1" in line[i]:
                        curr_line.append(line[i].split("=")[1])
                    elif "word=" in line[i] or "word2" in line[i]:
                        word = line[i].split("=")[1]
                        curr_line.append(word_map[word])
                    elif "path=" in line[i]:
                        path = line[i].split("=")[1]
                        curr_line.append(path_map[path])
                    elif "distance=" in line[i]:
                        distance = int(line[i].split("=")[1]) + 200
                        curr_line.append(distance)
            testing_features.append(curr_line)

        print(f'the length of testing features is {len(testing_features)}')
    else:
        sys.exit('invalid file type')


# scale each value to the range(0, 1) based on the min-max in the column
def scale(arr):
    for j in range(max_sentence_len):
        curr_col_max = max(arr[:, j])
        for i in range(arr.shape[0]):
            arr[i, j] = arr[i, j] / float(curr_col_max)
    return arr

# padding and scaling, and then fitting the model
# fit the model using features and labels
def fitModel(model_type):
    encodeFile("training")
    features = np.array([np.pad(sentence, (0, max_sentence_len - len(sentence)), 'constant') for sentence in training_features])
    print(f'the shape of training features is {features.shape}')
    features = features.astype('float')
    # features = scale(features)
    labels = np.array(training_labels, dtype=object).astype('float')
    print(f'the shape of training labels is {labels.shape}')

    if model_type == "naive_bayes":
        nb.fit(features, labels)
    elif model_type == "SVM":
        sgd.fit(features, labels)
    elif model_type == "decision_tree":
        decision_tree.fit(features, labels)
    elif model_type == "forest":
        forest.fit(features, labels)


def predict(model_type):
    encodeFile("testing")
    features = np.array(
        [np.pad(sentence, (0, max_sentence_len - len(sentence)), 'constant') for sentence in testing_features])

    print(f'the shape of testing feature is {features.shape}')
    features = features.astype('float')
    # features = scale(features)
    if model_type == "naive_bayes":
        label_pred = nb.predict(features)
    elif model_type == "SVM":
        label_pred = sgd.predict(features)
    elif model_type == "decision_tree":
        label_pred = decision_tree.predict(features)
    elif model_type == "forest":
        label_pred = forest.predict(features)

    print(f'the length of prediction output is {len(label_pred)}')
    write_output(model_type, testing_lines, label_pred)


def write_output(model_type, testing_lines, label_pred):
    if feature_selection == "stem":
        if model_type == "naive_bayes":
            out = open("../outputs/nb_stem.txt", 'w')
        elif model_type == "SVM":
            out = open("../outputs/svm_stem.txt", 'w')
        elif model_type == "decision_tree":
            out = open("../outputs/tree_stem.txt", 'w')
        elif model_type == "forest":
            out = open("../outputs/forest_stem.txt", 'w')
    elif feature_selection == "path":
        if model_type == "naive_bayes":
            out = open("../outputs/nb_path.txt", 'w')
        elif model_type == "SVM":
            out = open("../outputs/svm_path.txt", 'w')
        elif model_type == "decision_tree":
            out = open("../outputs/tree_path.txt", 'w')
        elif model_type == "forest":
            out = open("../outputs/forest_path.txt", 'w')
    elif feature_selection == "vector":
        if model_type == "naive_bayes":
            out = open("../outputs/nb_vector.txt", 'w')
        elif model_type == "SVM":
            out = open("../outputs/svm_vector.txt", 'w')
        elif model_type == "decision_tree":
            out = open("../outputs/tree_vector.txt", 'w')
        elif model_type == "forest":
            out = open("../outputs/forest_vector.txt", 'w')

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
    global feature_selection
    if "stem" in training_file and "stem" in testing_file:
        feature_selection += "stem"
    elif "path" in training_file and "path" in testing_file:
        feature_selection += "path"
    elif "vec" in training_file and "vec" in testing_file:
        feature_selection += "vector"
    else:
        sys.exit("invalid arguments provided, double checked please")

    constructMap(training_file, testing_file)
    fitModel("naive_bayes")
    predict("naive_bayes")
    # fitModel("SVM")
    # predict("SVM")
    fitModel("decision_tree")
    predict("decision_tree")
    fitModel("forest")
    predict("forest")



if __name__ == '__main__': sys.exit(main(sys.argv))
