import Orange
from pathlib import Path
import os
from typing import List, Dict

class MyCounter:
    def __init__(self, str_list: List[str]) -> None:
        self._state: Dict[str, int] = {}
        for value in str_list:
            if not self._state.get(value):
                self._state[value] = 1
            else:
                self._state[value] += 1
        self.__dict__ = self._state
        
    def __str__(self):
        return f'Counter({self.__dict__})'


credit_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'credit_data_regras.csv'

credit_base = Orange.data.Table(os.path.abspath(credit_base_path)) 

majority_learner = Orange.classification.MajorityLearner()

predictions = Orange.evaluation.testing.TestOnTestData(credit_base, credit_base, [majority_learner])

if __name__ == '__main__':
    print(Orange.evaluation.CA(predictions))
    print(MyCounter([str(registry.get_class()) for registry in credit_base]))
    from collections import Counter
    print(Counter([str(registry.get_class()) for registry in credit_base]))
    
    