from sklearn.preprocessing import StandardScaler
from categorical_attributes import X_census

scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

if __name__ == '__main__':
    print(X_census[0])
