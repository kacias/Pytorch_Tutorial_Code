from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from pandas import read_csv
import pandas as pd

df = read_csv('Test_Dataset.csv')
data = df.values
print(data[:, 2])

plt.scatter(data[:, 1], data[:,3], c="white", marker='o', edgecolors='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()


'''
x, y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.5, shuffle=True, random_state=0)
plt.scatter(x[:, 0], x[:,1], c="white", marker='o', edgecolors='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()
'''


kmeans = KMeans(n_clusters=2, init="random", n_init=10, max_iter=300, random_state=0)
pred = kmeans.fit_predict(data)
print(pred)


'''
tsne = TSNE(n_components=2)
y = tsne.fit_transform(data)
print(y)
plt.scatter(y[:,0], y[:,1], alpha=0.9, c=y[:,0], s=3, cmap='viridis')
plt.show()
'''


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(data, 'single')
labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

