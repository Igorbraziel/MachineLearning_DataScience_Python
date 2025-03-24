from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

census_path = Path(__file__).parent.parent / 'Base_de_dados' / 'census.csv'

census_base = pd.read_csv(census_path)

X_census = census_base.iloc[:, 0:14].values
y_census = census_base.iloc[:, 14].values

# Label Encoder
from sklearn.preprocessing import LabelEncoder

# workclass(1),education(3),marital-status(5),occupation(6),relationship(7),race(8),sex(9),native-country(13)

label_encoder_age = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital_status = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_native_country = LabelEncoder()

X_census[:, 1] = label_encoder_age.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital_status.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_native_country.fit_transform(X_census[:, 13])

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_census = scaler.fit_transform(X_census)

# Train and Test
from sklearn.model_selection import train_test_split

X_training, X_test, y_training, y_test = train_test_split(X_census, y_census, test_size=0.15, random_state=1) 