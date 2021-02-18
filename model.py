

# References
# https://machinelearningmastery.com/imbalanced-classification-with-the-adult-income-dataset/

from dowhy import CausalModel
import dowhy.datasets




##########################################################################

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]
#
# classifiers = [
#     # KNeighborsClassifier(3),
#     # SVC(kernel="linear", C=0.025),
#     # SVC(gamma=2, C=1),
#     # GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=2),
#     RandomForestClassifier(max_depth=8, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
#
# for name, clf in zip(names, classifiers):
#     clf.fit(X_train, y_train)
#     print("{}: {}".format(name, clf.score(X_test, y_test)))