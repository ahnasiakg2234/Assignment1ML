import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


breast_cancer = datasets.load_breast_cancer()
# Load the dataset
X = breast_cancer.data
y = breast_cancer.target
# partition the data into 80% training and 20% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_test = scaled.transform(X_test)

knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

precision = sklearn.metrics.precision_score(y_test, knn.predict(X_test))
print(f'Precision: {precision:.2f}')

recall = sklearn.metrics.recall_score(y_test, knn.predict(X_test))
print(f'Recall: {recall:.2f}')

f1_score = sklearn.metrics.f1_score(y_test, knn.predict(X_test))
print(f"F1 Score: {f1_score:.2f}")

confusion = sklearn.metrics.confusion_matrix(y_test, knn.predict(X_test))
print(f"Confusion Matrix:\n{confusion}")

display = ConfusionMatrixDisplay(confusion)
display.plot()
plt.show()