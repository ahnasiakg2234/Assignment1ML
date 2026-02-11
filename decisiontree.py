
import sklearn
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = sklearn.tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

tree.plot_tree(decision_tree)
plt.show()

accuracy = decision_tree.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

precision = sklearn.metrics.precision_score(y_test, decision_tree.predict(X_test))
print(f"Precision: {precision:.2f}")

recall = sklearn.metrics.recall_score(y_test, decision_tree.predict(X_test))
print(f"Recall: {recall:.2f}")

f1_score = sklearn.metrics.f1_score(y_test, decision_tree.predict(X_test))
print(f"F1 Score: {f1_score:.2f}")


confusion = sklearn.metrics.confusion_matrix(y_test, decision_tree.predict(X_test))
print(f"Confusion Matrix:\n{confusion}")

display = ConfusionMatrixDisplay(confusion)
display.plot()
plt.show()