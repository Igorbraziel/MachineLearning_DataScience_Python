from census_division_between_forecasters_and_class import X_census
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder

# Categorical attributes:
# 1            3            5            6           7        8    9        13
# workclass,education,marital-status,occupation,relationship,race,sex,native-country

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital_status = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship= LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_native_country = LabelEncoder()

if __name__ == '__main__':
    print(X_census[0])
    
X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital_status.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_native_country.fit_transform(X_census[:, 13])

if __name__ == '__main__':
    print(X_census[0])
    print(X_census.__class__.__name__)
    
# OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder_census = ColumnTransformer(
        [('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
        remainder='passthrough'
    )

X_census: ndarray = onehotencoder_census.fit_transform(X_census).toarray()

if __name__ == '__main__':
    print(X_census[0])
    print(X_census.shape)
    
