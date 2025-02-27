from pathlib import Path
import os
import Orange

census_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'census_regras.csv'

census_base = Orange.data.Table(os.path.abspath(census_base_path))

majority_learner = Orange.classification.MajorityLearner()

predictions = Orange.evaluation.testing.TestOnTestData(census_base, census_base, [majority_learner])

if __name__ == '__main__':
    print(Orange.evaluation.CA(predictions))
    from collections import Counter
    print(Counter(str(registry.get_class()) for registry in census_base))