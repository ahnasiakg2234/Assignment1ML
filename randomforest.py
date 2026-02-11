
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

accuracy = random_forest.score(X_test, y_test)
print("Accuracy:", accuracy)

precision = sklearn.metrics.precision_score(y_test, random_forest.predict(X_test))
print("Precision:", precision)

recall = sklearn.metrics.recall_score(y_test, random_forest.predict(X_test))
print("Recall:", recall)

f1_score = sklearn.metrics.f1_score(y_test, random_forest.predict(X_test))
print("F1 Score:", f1_score)

confusion = sklearn.metrics.confusion_matrix(y_test, random_forest.predict(X_test))
print("Confusion Matrix:\n", confusion)