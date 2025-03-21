from preparing_database import X_census, y_census

from imblearn.under_sampling import TomekLinks

tomek_links = TomekLinks(sampling_strategy='majority')

X_undersampling, y_undersampling = tomek_links.fit_resample(X_census, y_census)

print(X_census.shape, X_undersampling.shape)
print(y_census.shape, y_undersampling.shape)

#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

X_undersampling = onehot_encoder.fit_transform(X_undersampling).toarray()

print(X_undersampling.shape)

# Train and test
from sklearn.model_selection import train_test_split

X_training, X_test, y_training, y_test = train_test_split(X_undersampling, y_undersampling, test_size=0.2, random_state=1) 

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=400, criterion='entropy', random_state=1, min_samples_split=5, min_samples_leaf=2)
random_forest.fit(X_training, y_training)

# Accuracy Score
from sklearn.metrics import accuracy_score
previsions = random_forest.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, previsions))
