from credit_database import base_credit
import plotly.express as px    
    
# dropping the column with inconsistent values
base_credit2 = base_credit.drop('age', axis=1)
    
# dropping the records with inconsistent values
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

# Filling the inconsistent values manually (recommended)
age_mean = base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = age_mean

if __name__ == '__main__':
    print(base_credit.loc[base_credit['age'] < 0])
    print(base_credit2) 
    print(base_credit3.head(27))
    print(base_credit.head(27))
    scatter_graphic = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
    scatter_graphic.show()