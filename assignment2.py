import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import unittest

# Import Data
TrainData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
TrainData.head()
#print(TrainData.head)


Y = TrainData['meal']
X = TrainData.drop(columns=['meal','id','DateTime'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

#
model = RandomForestClassifier(n_estimators=100, random_state= 40)
modelFit = model.fit(x_train,y_train)

# view the feature scores

feature_scores = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(feature_scores)

class testCases(unittest.TestCase):
    def testValidModel(self):
        modelType = str(type(model))
        valid_model = False
        for i in ['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier']:
            if i in modelType:
                valid_model = True
                break
        self.assertTrue(valid_model, "Model type is not a valid classifier")

predict = model.predict(x_test)
acc = accuracy_score(y_test, predict)

print("Model accuracy is {}%.".format(acc*100))

# training accuracy vs test accuracy
train_accuracy = accuracy_score(y_train, model.predict(x_train))
test_accuracy = accuracy_score(y_test, predict)
print(f"\nTraining accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# PRED Test data
TestData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
xt = TestData.drop(columns = ['meal','id','DateTime'], axis =1)
pred = model.predict(xt) # still random forest
print(pred)


# Predict Random forerst
predTest2 = model.predict(xt) #model is linked from earlier

pred2 = pd.DataFrame(predTest2, columns=["predict_meal"])

pred2["predict_meal"] = pred2["predict_meal"].astype(int)

print("\nRandom Forest Predictions on test data:")
print(pred2.value_counts())

# Predict on the training data to get predictions of the same size as y_train
pred_train = model.predict(x_train)
cm = confusion_matrix(y_train, pred_train)

print('Confusion matrix\n\n', cm)

# Test Functions: Ensure the models are working as expected
def test_model_accuracy(model, x_test, y_test):
    pred = model.predict(x_test)
    return accuracy_score(y_test, pred)

rf_test_acc = test_model_accuracy(model, x_test, y_test)
dt_test_acc = test_model_accuracy(DecisionTreeClassifier(max_depth=10, min_samples_leaf=10).fit(x_train,y_train), x_test, y_test)

print(f"\nRandom Forest Test accuracy: {rf_test_acc * 100:.2f}%")
