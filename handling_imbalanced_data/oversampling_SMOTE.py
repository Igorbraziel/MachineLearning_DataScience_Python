from preparing_database import X_census, y_census

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')

X_oversampling, y_oversampling = smote.fit_resample(X_census, y_census)

print(X_census.shape, X_oversampling.shape)
print(y_census.shape, y_oversampling.shape)

# Train and test
from sklearn.model_selection import train_test_split

X_training, X_test, y_training, y_test = train_test_split(X_oversampling, y_oversampling, test_size=0.2, random_state=1) 

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=400, criterion='entropy', random_state=1, min_samples_split=5, min_samples_leaf=2)
random_forest.fit(X_training, y_training)

# Accuracy Score
from sklearn.metrics import accuracy_score, classification_report
previsions = random_forest.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, previsions))
print('Classification Report:', classification_report(y_test, previsions))
