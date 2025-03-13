import pandas as pd
from apyori import apriori

from pathlib import Path
from pprint import pprint

market_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'mercado.csv'

market_base = pd.read_csv(market_base_path, header=None)

rows_number = market_base.shape[0]
columns_number = market_base.shape[1]

transactions = []

for i in range(rows_number):
    transactions.append(
        [
            market_base.values[i, j] for j in range(columns_number)
        ]
    )
    
pprint(transactions)

