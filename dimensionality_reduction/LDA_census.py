from preparing_database import X_training, X_test, y_training, y_test

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)

X_lda_training = lda.fit_transform(X_training, y_training)
X_lda_test = lda.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=400, criterion='entropy', min_samples_leaf=2, min_samples_split=5)
random_forest.fit(X_lda_training, y_training)

previsions = random_forest.predict(X_lda_test)

from sklearn.metrics import accuracy_score

print('Accuracy Score:', accuracy_score(y_test, previsions))