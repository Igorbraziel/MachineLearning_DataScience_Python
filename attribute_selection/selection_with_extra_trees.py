from preparing_database import X_census, y_census

from sklearn.ensemble import ExtraTreesClassifier

selection = ExtraTreesClassifier()
selection.fit(X_census, y_census)

importances = selection.feature_importances_

if __name__ == '__main__':
    print(X_census.shape)
    print(importances)
    
indexes = []
threshold = 0.05
for i in range(len(importances)):
    importance_value = importances[i]
    if importance_value >= threshold:
        indexes.append(i)
    
if __name__ == '__main__':
    print(indexes)
    
X_census = X_census[:, indexes]

if __name__ == '__main__':
    print(X_census)
    
# Train and test
from sklearn.model_selection import train_test_split

X_training, X_test, y_training, y_test = train_test_split(X_census, y_census, test_size=0.2, random_state=1) 

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=400, criterion='entropy', random_state=1, min_samples_split=5, min_samples_leaf=2)
random_forest.fit(X_training, y_training)

# Accuracy Score
from sklearn.metrics import accuracy_score
previsions = random_forest.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, previsions))
    
