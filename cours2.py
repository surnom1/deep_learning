from sklearn import *
from sklearn.datasets import load_iris
from sklearn import cluster
import matplotlib.pyplot as plt
import pandas as pd
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df = iris_df.drop(['sepal length (cm)','sepal width (cm)'],axis=1)

kmeansiris = cluster.KMeans(n_clusters=3)

kmeansiris.fit(iris_df)

plt.scatter(iris_df.iloc[:, 0], iris_df.iloc[:, 1], c=iris.target)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Iris Data Clustering')
plt.show()