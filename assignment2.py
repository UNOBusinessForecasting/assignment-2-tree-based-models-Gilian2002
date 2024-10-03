import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import
TrainData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')

#train
Y = TrainData['meal']
X = TrainData.drop(columns=['meal', 'id', 'DateTime', 'Total', 'Discounts'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#modell
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced")
modelFit = model.fit(x_train, y_train)

predict = model.predict(x_test)
acc = accuracy_score(y_test, predict)

print("Model accuracy is {}%.".format(acc*100))
