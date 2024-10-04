import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import
TrainData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
TrainData.head()
print(TrainData.head)

#
from sklearn.model_selection import train_test_split

Y = TrainData['meal']
X = TrainData.drop(columns=['meal','id','DateTime'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model

model = RandomForestClassifier(n_estimators=100, random_state= 42, class_weight = "balanced")
modelFit = model.fit(x_train,y_train)

# Pred.
predict = model.predict(x_test)
acc = accuracy_score(y_test, predict)

print("Model accuracy is {}%.".format(acc*100))

# PRED Test data / import
TestData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
xt = TestData.drop(columns = ['meal','id','DateTime'], axis =1)
pred = model.predict(xt)
print(pred)


# Predict Random forerst
predTest2 = model.predict(xt) #model is linked from earlier

pred2 = pd.DataFrame(predTest2, columns=["predict_meal"])

pred2["predict_meal"] = pred2["predict_meal"].astype(int)

print("\nRandom Forest Predictions on test data:")
print(pred2.value_counts())


