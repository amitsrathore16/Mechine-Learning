import numpy as np
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('news.csv')
y = dataset.label

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], y, test_size = 0.2, random_state = 7)

# Initializing TfidfVectorizer and fitting traing set
from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(stop_words='english', max_df = 0.7)
tvf_train = tfv.fit_transform(x_train)
tvf_test = tfv.transform(x_test)

#Fitting training set to PassiveAggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter= 50)
classifier.fit(tvf_train, y_train)

# Predicting test set result
y_pred = classifier.predict(tvf_test)

# Confusion matrix amd Accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)