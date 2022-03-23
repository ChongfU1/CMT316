import sklearn
import nltk
import pandas as pd
import numpy as np
import os
import jieba
import operator
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from matplotlib.pyplot import plot
nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed
nltk.download('omw-1.4') # If needed

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
              # Precise pattern slice and dice article
             # One word_list for one article
            #data_list.append(word_list)
            if folder == "business":
                category=0
            elif folder == "entertainment":
                category=1
            elif folder == "politics":
                category=2
            elif folder == "sport":
                category=3
            elif folder == "tech":
                category=4
            data_list.append([raw,category])
    return pd.DataFrame(data_list, columns=["content", "category"])
raw_data=[]
raw_data.append(text_processor(path))
full_set = pd.concat(raw_data, ignore_index=True)
full_set_x = full_set.iloc[: , :-1]
full_set_y = full_set.iloc[: , -1]
train_X, test_X, train_Y, test_Y = train_test_split(full_set_x, full_set_y, test_size=0.2, random_state=1)
tfidf_vector = TfidfVectorizer()
# Learn vocabulary and idf from training set
tfidf_vector.fit(train_X["content"])
# Transform train and test input documents to document-term matrix
tfidf_train_x = tfidf_vector.transform(train_X["content"])
tfidf_test_x  = tfidf_vector.transform(test_X["content"])

svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf.fit(tfidf_train_x, train_Y.to_numpy())
# Test with test data
predictions = svm_clf.predict(tfidf_test_x)
tfidf_test_y = test_Y.to_numpy()

precision = precision_score(tfidf_test_y, predictions, average='macro')
recall = recall_score(tfidf_test_y, predictions, average='macro')
f1 = f1_score(tfidf_test_y, predictions, average='macro')
accuracy = accuracy_score(tfidf_test_y, predictions)
print("Precision: " + str(round(precision, 3)))
print("Recall: " + str(round(recall, 3)))
print("F1-Score: " + str(round(f1, 3)))
print("Accuracy: " + str(round(accuracy, 3)))
print(classification_report(y_true, y_pred))