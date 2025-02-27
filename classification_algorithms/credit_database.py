import pandas as pd

from pathlib import Path
import os

current_path = Path(os.path.abspath('.'))
base_credit_path = Path(os.path.join(current_path.resolve(), 'Base_de_dados', 'credit_data.csv'))

base_credit = pd.read_csv(base_credit_path.resolve())

if __name__ == '__main__':
    # print(base_credit.head())
    # print(base_credit.tail())
    print(base_credit.describe())

    # Client with the highest income
    print(base_credit[base_credit['income'] >= 69995.685578])

    # Client with smallest debt
    print(base_credit[base_credit['loan'] <= 1.377630])
