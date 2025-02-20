import pickle

from pathlib import Path
import os

from training_and_test_base import X_credit_training, Y_credit_training, X_credit_test, Y_credit_test
from training_and_test_base import X_census_training, Y_census_training, X_census_test, Y_census_test

root_path = Path(__file__).parent.parent

pickle_credit_path = root_path / 'credit.pkl'
pickle_census_path = root_path / 'census.pkl'

with open(pickle_credit_path, 'wb') as file_:
    pickle.dump([X_credit_training, Y_credit_training, X_credit_test, Y_credit_test], file_)
    
with open(pickle_census_path, 'wb') as file_:
    pickle.dump([X_census_training, Y_census_training, X_census_test, Y_census_test], file_)
    