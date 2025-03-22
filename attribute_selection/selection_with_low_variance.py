from preparing_database import X_census, census_base
import numpy as np

columns_number = X_census.shape[1]
    
if __name__ == '__main__':
    for i in range(columns_number):
        variance = X_census[:, i].var()
        print(variance)
        
from sklearn.feature_selection import VarianceThreshold

threshold = 0.05
selection = VarianceThreshold(threshold=threshold)

X_census = selection.fit_transform(X_census)

if __name__ == '__main__':
    print(X_census.shape)
    
indexes = [index for index in range(columns_number + 1) if index not in np.where(selection.variances_ > threshold)[0] and index != 14]

new_census_base = census_base.drop(columns=census_base.columns[indexes], axis=1)

if __name__ == '__main__':
    print(new_census_base)

X_census = new_census_base.iloc[:, 0:5].values
y_census = new_census_base.iloc[:, 5].values

# Label Encoder
from sklearn.preprocessing import LabelEncoder

# workclass(1),education(3),marital-status(5),occupation(6),relationship(7),race(8),sex(9),native-country(13)

label_encoder_education = LabelEncoder()
label_encoder_marital_status = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_sex = LabelEncoder()


X_census[:, 0] = label_encoder_education.fit_transform(X_census[:, 0])
X_census[:, 1] = label_encoder_marital_status.fit_transform(X_census[:, 1])
X_census[:, 2] = label_encoder_occupation.fit_transform(X_census[:, 2])
X_census[:, 3] = label_encoder_relationship.fit_transform(X_census[:, 3])
X_census[:, 4] = label_encoder_sex.fit_transform(X_census[:, 4])

# Standardization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_census = scaler.fit_transform(X_census)

#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')

X_census = onehot_encoder.fit_transform(X_census)

if __name__ == '__main__':
    print(X_census.shape)

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


