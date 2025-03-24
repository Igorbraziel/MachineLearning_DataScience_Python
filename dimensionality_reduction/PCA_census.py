from preparing_database import X_training, X_test, y_training, y_test

from sklearn.decomposition import PCA

pca = PCA(n_components=7)

X_pca_training = pca.fit_transform(X_training)
X_pca_test = pca.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=400, criterion='entropy', min_samples_leaf=2, min_samples_split=5)
random_forest.fit(X_pca_training, y_training)

previsions = random_forest.predict(X_pca_test)

from sklearn.metrics import accuracy_score

print('Accuracy Score:', accuracy_score(y_test, previsions))