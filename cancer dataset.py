import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle

dataset = pd.read_csv('dataset/cancer - cancer.csv')

print(dataset)

dataset.drop(['Sample code number'],axis=1,inplace=True)

print(dataset)

x = dataset.drop(columns='Class', axis=1)
y = dataset['Class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=0)


Classifier= LogisticRegression(random_state=0)
Classifier.fit(x_train,y_train)

y_pred = Classifier.predict(x_test)
print(y_pred)



cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=Classifier,X=x_train,y=y_train,cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

filename = 'cancer_model.sav'
pickle.dump(Classifier, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open('cancer_model.sav', 'rb'))
for column in x.columns:
  print(column)





