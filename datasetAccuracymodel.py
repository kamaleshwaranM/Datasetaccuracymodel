import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
df= pd.read_csv("C:/Users/Pradeep_NG/Desktop/datasetAccuracyModel/dataset.csv")

#declaring dependent and independent variables
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#filling na values
#print(df.isnull().sum())
df = df.fillna(df.mean())
#print("\n NULL Values Count after filling \n", df.isnull().sum())

#correlation
#sns.heatmap(df.corr())

#splitting datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#xgboost classifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
xg=accuracy_score(y_test, preds)
print("Accuracy of XGB: ",xg )

"""
print("No of Nan values for X_train =",np.isnan(X_train).sum())
print("No of Nan values for y_train =",np.isnan(y_train).sum())
print("No of Nan values for X_test =",np.isnan(X_test).sum())
print("No of Nan values for y_test =",np.isnan(y_test).sum())
"""

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier

cl = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 5)
model2 =cl.fit(X_train, y_train)
y_pred = model2.predict(X_test)
dt=accuracy_score(y_test, y_pred)
print("Acurracy of DCT: ", dt)

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
result = accuracy_score(y_test,y_pred)
print("Accuracy of KNN classifier:",result)

#naive bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
nb = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of Naive Bayes:", nb)





