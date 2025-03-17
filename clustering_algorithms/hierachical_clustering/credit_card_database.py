import pandas as pd

from pathlib import Path

credit_card_base_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'credit_card_clients.csv'

credit_card_base = pd.read_csv(credit_card_base_path, header=1)

credit_card_base['BILL_TOTAL'] = credit_card_base['BILL_AMT1'] + credit_card_base['BILL_AMT2'] + credit_card_base['BILL_AMT3'] + credit_card_base['BILL_AMT4'] + credit_card_base['BILL_AMT5'] + credit_card_base['BILL_AMT6']

print(credit_card_base)

X_card = credit_card_base.iloc[:, [1, 25]].values

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

dendrogram_plot = dendrogram(linkage(X_card, method='ward'))
plt.title('Dendrogram')
plt.xlabel('People')
plt.ylabel('Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering

hierachical_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')

labels = hierachical_clustering.fit_predict(X_card)

import plotly.express as px

plot = px.scatter(x=X_card[:, 0], y=X_card[:, 1], color=labels)
plot.show()