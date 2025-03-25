from pathlib import Path
import pandas as pd
import plotly.express as px

credit_data_path = Path(__file__).parent.parent / 'Base_de_dados' / 'credit_data.csv'

credit_data_base = pd.read_csv(credit_data_path)
    
credit_data_base.dropna(inplace=True)

# Loan x Age
plot = px.scatter(x=credit_data_base['loan'], y=credit_data_base['age'])
plot.show()

if __name__ == '__main__':
    print(credit_data_base.isnull().sum())

outliers_age = credit_data_base[credit_data_base['age'] < 0]

plot = px.scatter(x=credit_data_base['loan'], y=credit_data_base['income'])
plot.show()

outliers_loan = credit_data_base[credit_data_base['loan'] > 13200]

if __name__ == '__main__':
    print(outliers_age, outliers_loan, sep='\n\n')