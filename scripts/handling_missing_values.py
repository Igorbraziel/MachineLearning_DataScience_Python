from handling_inconsistent_values import base_credit
import pandas as pd

if __name__ == '__main__':
    print(base_credit.isnull())
    print(base_credit.isnull().sum())
    print(base_credit.loc[pd.isnull(base_credit['age'])])
    
base_credit.fillna({'age': base_credit['age'].mean()}, inplace=True)

if __name__ == '__main__':
    print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])