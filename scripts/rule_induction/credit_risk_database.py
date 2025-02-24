import Orange
from pathlib import Path
import os

credit_risk_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'risco_credito_regras.csv'

credit_risk_base = Orange.data.Table(os.path.abspath(credit_risk_path))

if __name__ == '__main__':
    print(credit_risk_base)
    
cn2 = Orange.classification.rules.CN2Learner()
credit_risk_rules = cn2(credit_risk_base)

if __name__ == '__main__':
    print()
    for rule in credit_risk_rules.rule_list:
        print(rule)
        
predictions = credit_risk_rules([
    ['boa', 'alta', 'nenhuma', 'acima_35'],
    ['ruim', 'alta', 'adequada', '0_15']
])

if __name__ == '__main__':
    print()
    for value in predictions:
        print(credit_risk_base.domain.class_var.values[value])
