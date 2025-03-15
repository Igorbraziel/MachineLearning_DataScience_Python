# Is the same process but using more attributes

import pandas as pd

from pathlib import Path

credit_card_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'credit_card_clients.csv'

credit_card_base = pd.read_csv(credit_card_base_path, header=1)

credit_card_base['BILL_TOTAL'] = credit_card_base['BILL_AMT1'] + credit_card_base['BILL_AMT2'] + credit_card_base['BILL_AMT3'] + credit_card_base['BILL_AMT4'] + credit_card_base['BILL_AMT5'] + credit_card_base['BILL_AMT6']

print(credit_card_base)

X_card = credit_card_base.iloc[:, [1, 2, 3, 4, 5, 25]].values

from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_card = X_scaler.fit_transform(X_card)

from sklearn.cluster import KMeans

# test with WCSS
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_card)
    wcss.append(kmeans.inertia_)
    
# Analysis of WCSS
import plotly.express as px
graphic = px.line(x=range(1, 11), y=wcss)
graphic.show()

kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(X_card)

# Converting the X for 2 variables
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_card = pca.fit_transform(X_card)

graphic1 = px.scatter(x=X_card[:, 0], y=X_card[:, 1], color=labels)
graphic1.show()

import numpy as np
clients_list = np.column_stack((credit_card_base, labels))
clients_list = clients_list[clients_list[:, 26].argsort()]
print(clients_list)
