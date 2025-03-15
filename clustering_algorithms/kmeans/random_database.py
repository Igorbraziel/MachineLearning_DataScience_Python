from sklearn.datasets import make_blobs

X_random, y_random = make_blobs(n_samples=200, centers=5, random_state=1, n_features=2)

print(X_random.shape, y_random.shape, sep='\n')

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(X_random)

labels = kmeans.predict(X_random)

centroids = kmeans.cluster_centers_

import plotly.express as px
import plotly.graph_objects as go

graphic1 = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=labels)
graphic2 = px.scatter(x=centroids[:, 0], y=centroids[:, 1], size=[12 for _ in range(len(centroids))])
graphic3 = go.Figure(data = graphic1.data + graphic2.data)
graphic3.show()