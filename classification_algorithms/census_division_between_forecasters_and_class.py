from census_database import base_census

X_census = base_census.iloc[:, 0:14].values
Y_census = base_census.iloc[:, 14].values

if __name__ == '__main__':
    print(X_census)
    print(Y_census)