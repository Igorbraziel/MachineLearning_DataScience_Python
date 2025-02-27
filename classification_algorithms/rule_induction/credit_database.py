import Orange
from pathlib import Path
import os


credit_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'credit_data_regras.csv'

credit_base = Orange.data.Table(os.path.abspath(credit_base_path)) 

if  __name__ == '__main__':
    print(credit_base.domain)
    
splitted_base = Orange.evaluation.testing.sample(credit_base, n=0.25)

test_base, training_base = splitted_base

if  __name__ == '__main__':
    print(test_base.__len__(), training_base.__len__())
    
cn2 = Orange.classification.rules.CN2Learner()
credit_rules = cn2(credit_base)

if  __name__ == '__main__':
    for rule in credit_rules.rule_list:
        print(rule)
        
predictions = Orange.evaluation.testing.TestOnTestData(training_base, test_base, [lambda testdata: credit_rules])

if  __name__ == '__main__':
    print(Orange.evaluation.CA(predictions)[0])