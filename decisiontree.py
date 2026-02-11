
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split


breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = sklearn.tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

decision_tree_max_depth = sklearn.tree.DecisionTreeClassifier(max_depth=5)
decision_tree_max_depth.fit(X_train, y_train)

accuracy = decision_tree.score(X_test, y_test)
accuracy2 = decision_tree_max_depth.score(X_test, y_test)
print("Accuracy:", accuracy)
print("Accuracy (max_depth=5):", accuracy2)

precision = sklearn.metrics.precision_score(y_test, decision_tree.predict(X_test))
print("Precision:", precision)
precision2 = sklearn.metrics.precision_score(y_test, decision_tree_max_depth.predict(X_test))
print("Precision (max_depth=5):", precision2)

recall = sklearn.metrics.recall_score(y_test, decision_tree.predict(X_test))
print("Recall:", recall)
recall2 = sklearn.metrics.recall_score(y_test, decision_tree_max_depth.predict(X_test))
print("Recall (max_depth=5):", recall2)

f1_score = sklearn.metrics.f1_score(y_test, decision_tree.predict(X_test))
print("F1 Score:", f1_score)
f1_score2 = sklearn.metrics.f1_score(y_test, decision_tree_max_depth.predict(X_test))
print("F1 Score (max_depth=5):", f1_score2)

confusion = sklearn.metrics.confusion_matrix(y_test, decision_tree.predict(X_test))
print("Confusion Matrix:\n", confusion)
confusion2 = sklearn.metrics.confusion_matrix(y_test, decision_tree_max_depth.predict(X_test))
print("Confusion Matrix (max_depth=5):\n", confusion2)