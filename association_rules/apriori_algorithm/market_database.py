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
            str(market_base.values[i, j]) for j in range(columns_number)
        ]
    )
    
pprint(transactions)

apriori_rules = apriori(transactions=transactions, min_support=0.3, min_confidence=0.8, min_lift=2)

apriori_results = list(apriori_rules)

A = [] # A list for the IF's of the rules
B = [] # A list for the SO's of the rules
support = []
confidence = []
lift = []

for result in apriori_results:
    support_value = result[1]
    result_rules = result[2]
    
    for result_rule in result_rules:
        a = list(result_rule[0])
        b = list(result_rule[1])
        confidence_value = result_rule[2]
        lift_value = result_rule[3]
        
        A.append(a)
        B.append(b)
        support.append(support_value)
        confidence.append(confidence_value)
        lift.append(lift_value)
        
rules_data_frame = pd.DataFrame({
    'A': A,
    'B': B,
    'Support': support,
    'Confidence': confidence,
    'Lift': lift,    
})

rules_data_frame = rules_data_frame.sort_values(
    by='Lift',
    ascending=False
)

print(rules_data_frame)
