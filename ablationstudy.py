import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_test = scaled.transform(X_test)

#Modification of key hyperparameters, see how nearest neighbors changes the data
knn = KNeighborsClassifier(n_neighbors= 25)
knn.fit(X_train, y_train)

# Evaluate the KNN model with modified key hyperparameter n_neighbors
print("KNN Classifier (n_neighbors=25):")
accKNN = knn.score(X_test, y_test)
print("Accuracy:", accKNN)

precKNN = sklearn.metrics.precision_score(y_test, knn.predict(X_test))
print("Precision:", precKNN)

recKNN = sklearn.metrics.recall_score(y_test, knn.predict(X_test))
print("Recall:", recKNN)

f1_scoreKNN = sklearn.metrics.f1_score(y_test, knn.predict(X_test))
print("F1 Score:", f1_scoreKNN)

confusionKNN = sklearn.metrics.confusion_matrix(y_test, knn.predict(X_test))
print("Confusion Matrix:\n", confusionKNN)

#Experimenting with max_depth for decision tree
decision_tree_max_depth = sklearn.tree.DecisionTreeClassifier(max_depth=5)
decision_tree_max_depth.fit(X_train, y_train)

#Evaluate the Decision Tree model with modified key hyperparameter max_depth
print("\nDecision Tree Classifier (max_depth=5):")
accDT = decision_tree_max_depth.score(X_test, y_test)
print("Accuracy (max_depth=5):", accDT)

precDT = sklearn.metrics.precision_score(y_test, decision_tree_max_depth.predict(X_test))
print("Precision (max_depth=5):", precDT)

recDT = sklearn.metrics.recall_score(y_test, decision_tree_max_depth.predict(X_test))
print("Recall (max_depth=5):", recDT)

f1_scoreDT = sklearn.metrics.f1_score(y_test, decision_tree_max_depth.predict(X_test))
print("F1 Score (max_depth=5):", f1_scoreDT)

confusionDT = sklearn.metrics.confusion_matrix(y_test, decision_tree_max_depth.predict(X_test))
print("Confusion Matrix (max_depth=5):\n", confusionDT)

#Exploring the effect of max_depth and min_samples_split for random forests
random_forest_max_depth = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=5)
random_forest_max_depth.fit(X_train, y_train)
#Evaluate the random forest model using max_depth
print("\nRandom Forest Classifier with (max_depth=5):")
accRFMax = random_forest_max_depth.score(X_test, y_test)
print("Accuracy (max_depth=5):", accRFMax)

precRFMax = sklearn.metrics.precision_score(y_test, random_forest_max_depth.predict(X_test))
print("Precision (max_depth=5):", precRFMax)

recRFMax = sklearn.metrics.recall_score(y_test, random_forest_max_depth.predict(X_test))
print("Recall (max_depth=5):", recRFMax)

f1_scoreRFMax = sklearn.metrics.f1_score(y_test, random_forest_max_depth.predict(X_test))
print("F1 Score (max_depth=5):", f1_scoreRFMax)

confusionRFMax = sklearn.metrics.confusion_matrix(y_test, random_forest_max_depth.predict(X_test))
print("Confusion Matrix (max_depth=5):\n", confusionRFMax)

#Exploring the effect of using min_samples_split
random_forest_min_samples_split = sklearn.ensemble.RandomForestClassifier(n_estimators=100, min_samples_split=15)
random_forest_min_samples_split.fit(X_train, y_train)
#Evaluate the model
print("\nRandom Forest Classifier with (min_samples_split=15):")
accRFMin = random_forest_min_samples_split.score(X_test, y_test)
print("Accuracy (min_samples_split=15):", accRFMin)

precRFMin = sklearn.metrics.precision_score(y_test, random_forest_min_samples_split.predict(X_test))
print("Precision (min_samples_split=15):", precRFMin)

recRFMin = sklearn.metrics.recall_score(y_test, random_forest_min_samples_split.predict(X_test))
print("Recall (min_samples_split=15):", recRFMin)

f1_scoreRFMin = sklearn.metrics.f1_score(y_test, random_forest_min_samples_split.predict(X_test))
print("F1 Score (min_samples_split=15):", f1_scoreRFMin)

confusionRFMin = sklearn.metrics.confusion_matrix(y_test, random_forest_min_samples_split.predict(X_test))
print("Confusion Matrix (min_samples_split=15):\n", confusionRFMin)
