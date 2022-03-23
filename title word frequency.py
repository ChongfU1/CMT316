import os
import nltk
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random
from nltk.corpus import stopwords
import operator

from sklearn.svm import LinearSVC
import jieba
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
nltk.download('wordnet')
nltk.download('punkt')
clf = LinearSVC()
svm = CalibratedClassifierCV(clf)
# Text processing --> Generate training set Test set Word frequency set
path = os.getcwd() + "/bbc"
def text_processor(text_path, test_size=0.2):
    folder_list = os.listdir(text_path)
    data_list = []  # Each element is an article
    class_list = []  # Corresponds to the category of each article
    # A loop reads a category of folders
    for folder in folder_list:
        if folder.endswith('.TXT'):
            continue
        new_folder_path = os.path.join(text_path, folder)  # List of categories
        #files = random.sample(os.listdir(new_folder_path),10)
        files = os.listdir(new_folder_path)
        # A loop to read an article
        for file in files:
            #print(file)
            with open(os.path.join(new_folder_path, file), 'r', encoding='UTF-8',errors='ignore') as fp:
                raw = fp.read()
            word_cut = jieba.cut(raw, cut_all=True)  # Precise pattern slice and dice article
            word_list = list(word_cut)  # One word_list for one article
            data_list.append(word_list)
            if folder == "business":
                class_list.append(0)
            elif folder == "entertainment":
                class_list.append(1)
            elif folder == "politics":
                class_list.append(2)
            elif folder == "sport":
                class_list.append(3)
            elif folder == "tech":
                class_list.append(4)
    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)  # Upset the order
    index = int(len(data_class_list) * test_size) + 1  # The training ratio is 8:2

    train_list = data_class_list[index:]
    test_list = data_class_list[:index]

    train_data_list, train_class_list = zip(*train_list)  # (word_list_one[],...), (Sports,...)
    test_data_list, test_class_list = zip(*test_list)

    # Statistical word frequency all_words_dict{"key_word_one":100, "key_word_two":200, ...}
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.get(word) != None:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # Sort by value in descending order
    all_words_list = list(list(zip(*all_words_tuple_list))[0])  # all_words_list[word_one, word_two, ...]
    #print("Test text real labels：",test_class_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list
all_words_list, train_data_list, test_data_list, train_class_list, test_class_list=text_processor((path))

stopwords = nltk.corpus.stopwords.words('english')
new_stopwords = ["/", ".","@", "\n", ",", "'s", "``", "''", "'", "n't", "%", "-", "$", "(", ")", ":", ";"]
stopwords.extend(new_stopwords)
# Selecting feature words
def words_dict(all_words_list, deleteN):
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 700:  # Dimensional max. 10
            break
        # Non-numeric Non-stop words Length between 1 and 4
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords and 1 < len(
                all_words_list[t]) < 3:
            feature_words.append(all_words_list[t])
            n += 1
    print("Machine learning text features：",feature_words)
    return feature_words
feature_words= words_dict(all_words_list,0)
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing the sklearn library
# Text Features
def text_features(train_data_list, test_data_list, feature_words):
    def text_feature_(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_feature_(text, feature_words) for text in train_data_list]
    test_feature_list = [text_feature_(text, feature_words) for text in test_data_list]
    print("Test document word frequency vectors：",test_feature_list,"\n")
    return train_feature_list, test_feature_list

train_feature_list, test_feature_list=text_features(train_data_list,test_data_list,feature_words)
def text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
   classifier = svm.fit(train_feature_list, train_class_list)
   y_pred = classifier.predict(test_feature_list)
   y_true = test_class_list


   precision = precision_score(y_true, y_pred, average='macro')
   recall = recall_score(y_true, y_pred, average='macro')
   f1 = f1_score(y_true, y_pred, average='macro')
   accuracy = accuracy_score(y_true, y_pred)

   print("Precision: " + str(round(precision, 3)))
   print("Recall: " + str(round(recall, 3)))
   print("F1-Score: " + str(round(f1, 3)))
   print("Accuracy: " + str(round(accuracy, 3)))
   print(classification_report(y_true, y_pred))

text_classifier(train_feature_list, test_feature_list, train_class_list, test_class_list)