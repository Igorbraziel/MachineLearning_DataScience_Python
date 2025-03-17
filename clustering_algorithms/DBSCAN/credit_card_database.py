import pandas as pd

from pathlib import Path

credit_card_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'credit_card_clients.csv'

credit_card_base = pd.read_csv(credit_card_base_path, header=1)

credit_card_base['BILL_TOTAL'] = credit_card_base['BILL_AMT1'] + credit_card_base['BILL_AMT2'] + credit_card_base['BILL_AMT3'] + credit_card_base['BILL_AMT4'] + credit_card_base['BILL_AMT5'] + credit_card_base['BILL_AMT6']

X_card = credit_card_base.iloc[:, [1, 25]].values

from sklearn.cluster import DBSCAN

dbscan_credit_card = DBSCAN(eps=0.37, min_samples=5)
labels = dbscan_credit_card.fit_predict(X_card)

import numpy as np

print(np.unique(labels, return_counts=True))

import plotly.express as px

plot = px.scatter(x=X_card[:, 0], y=X_card[:, 1], color=labels)
plot.show()