from handling_missing_values import base_credit

X_credit = base_credit.iloc[:, 1:4].values
Y_credit = base_credit.iloc[:, 4].values

if __name__ == '__main__':
    print(X_credit)
    print(Y_credit)
