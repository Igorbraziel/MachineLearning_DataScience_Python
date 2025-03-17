from sklearn.datasets import make_moons
import plotly.express as px

X_random, y_random = make_moons(n_samples=1500, noise=0.09)

moon_plot = px.scatter(x=X_random[:, 0], y=X_random[:, 1])
moon_plot.show()

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(X_random)
plot = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=labels)
plot.show()


hierarchical_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = hierarchical_clustering.fit_predict(X_random)
plot = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=labels)
plot.show()

dbscan = DBSCAN(eps=0.1)
labels = dbscan.fit_predict(X_random)
plot = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=labels)
plot.show()

