from pathlib import Path
import os
import Orange

census_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'census_regras.csv'

census_base = Orange.data.Table(os.path.abspath(census_base_path))
census_base = census_base[:round(len(census_base)/10)]

splitted_base = Orange.evaluation.testing.sample(census_base, n=0.25)

test_census_base, training_census_base = splitted_base

cn2 = Orange.classification.rules.CN2Learner()

census_rules = cn2(training_census_base)

predictions = Orange.evaluation.testing.TestOnTestData(
    training_census_base, test_census_base,
    [lambda testdata: census_rules]
)

if __name__ == '__main__':
    for rule in census_rules.rule_list:
        print(rule)
    print(Orange.evaluation.CA(predictions))