from attribute_scaling import X_credit, Y_credit
from census_attribute_scaling import X_census
from census_division_between_forecasters_and_class import Y_census

from sklearn.model_selection import train_test_split

X_credit_training, X_credit_test, Y_credit_training, Y_credit_test = train_test_split(X_credit, Y_credit, test_size=0.25, random_state=0)

X_census_training, X_census_test, Y_census_training, Y_census_test = train_test_split(X_census, Y_census, test_size=0.15, random_state=0)


if __name__ == '__main__':
    print(X_credit_training.shape)
    print(Y_credit_training.shape)
    
    print(X_credit_test.shape)
    print(Y_credit_test.shape)
    
    print()
    
    print(X_census_training.shape)
    print(Y_census_training.shape)
    
    print(X_census_test.shape)
    print(Y_census_test.shape)
