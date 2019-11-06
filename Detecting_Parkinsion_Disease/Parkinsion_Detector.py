import numpy as np
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('parkinsons.data')
features = dataset.loc[:, dataset.columns!='status'].values[:, 1:]
lables = dataset.loc[:, 'status'].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = lables

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)

# Fitting XgBoost to training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix and getting accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuries = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuries.mean()
accuries.std()