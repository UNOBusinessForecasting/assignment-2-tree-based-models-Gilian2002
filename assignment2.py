import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Import Data
TrainData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
TrainData.head()

# Split Data
Y = TrainData['meal']
X = TrainData.drop(columns=['meal', 'id', 'DateTime'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=40)
modelFit = model.fit(x_train, y_train)

# Feature importance
feature_scores = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print("\nFeature scores:\n", feature_scores)

# Model Predictions and Accuracy
predict = model.predict(x_test)
acc = accuracy_score(y_test, predict)
print(f"\nRandom Forest Model accuracy is {acc * 100:.2f}%")

# Overfitting Check: Compare training accuracy vs test accuracy
train_accuracy = accuracy_score(y_train, model.predict(x_train))
test_accuracy = accuracy_score(y_test, predict)
print(f"\nTraining accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# If training accuracy is much higher than test accuracy, the model may be overfitting
if train_accuracy - test_accuracy > 0.1:
    print("\nWarning: The model may be overfitting!")
else:
    print("\nThe model does not show signs of overfitting.")

# Predict on Test Data
TestData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
check_data_integrity(TestData)

xt = TestData.drop(columns=['meal', 'id', 'DateTime'], axis=1)
predTest2 = model.predict(xt)
pred2 = pd.DataFrame(predTest2, columns=["predict_meal"])

pred2["predict_meal"] = pred2["predict_meal"].astype(int)
print("\nRandom Forest Predictions on test data:")
print(pred2.value_counts())

# Predict with Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10).fit(x_train, y_train)
predTest3 = dt_model.predict(xt)
pred3 = pd.DataFrame(predTest3, columns=["predict_meal"])

pred3["predict_meal"] = pred3["predict_meal"].astype(int)
print("\nDecision Tree Predictions on test data:")
print(pred3.value_counts())

# Confusion Matrix for the training data (Random Forest)
pred_train = model.predict(x_train)
cm = confusion_matrix(y_train, pred_train)
print('\nConfusion matrix for Random Forest on training data:\n', cm)

# Test Functions: Ensure the models are working as expected
def test_model_accuracy(model, x_test, y_test):
    pred = model.predict(x_test)
    return accuracy_score(y_test, pred)

rf_test_acc = test_model_accuracy(model, x_test, y_test)
dt_test_acc = test_model_accuracy(dt_model, x_test, y_test)

print(f"\nRandom Forest Test accuracy: {rf_test_acc * 100:.2f}%")
print(f"Decision Tree Test accuracy: {dt_test_acc * 100:.2f}%")
