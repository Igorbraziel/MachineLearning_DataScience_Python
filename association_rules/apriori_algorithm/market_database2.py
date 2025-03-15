import pandas as pd
from apyori import apriori

from pathlib import Path
from pprint import pprint

market_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'mercado2.csv'

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

# For Products that are sold 4 times a day (in a week)
min_support = (4 * 7) / market_base.shape[0]

apriori_rules = apriori(transactions=transactions, min_support=min_support, min_confidence=0.2, min_lift=3)

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
        a = [value for value in a if value != 'nan']
        b = list(result_rule[1])
        b = [value for value in b if value != 'nan']
        confidence_value = result_rule[2]
        lift_value = result_rule[3]
        
        if a not in A or b not in B:
            A.append(a)
            B.append(b)
            support.append(support_value)
            confidence.append(confidence_value)
            lift.append(lift_value)
        else:
            flag = False
            a_occurrences = [index for index in range(len(A)) if A[index] == a]
            b_occurrences = [index for index in range(len(B)) if B[index] == b]        
                    
            for index in a_occurrences:
                if index in b_occurrences:
                    flag = True
            
            if flag == False:
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
