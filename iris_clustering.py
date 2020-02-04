
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Data Centering
def centering(data):
    return data.apply(np.log)


# Standardization
def standardization(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


# Decorrelation
def decorrelation(data):
    pca = PCA()
    pca.fit(data)
    return pca.transform(data)


# ReStandardization
def restandardization(data):
    pass


# Feature Reduction
def feature_variance_check(data):
    pca = PCA()
    pca.fit(data)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()


def feature_reduction(data, num_com=2):
    pca = PCA(n_components=num_com)
    pca.fit(data)
    pca_features = pca.transform(data)
    print(pca_features.shape)
    return pca_features


# Finding the optimal number of clusters
def optimal_clusters(data):
    ks = range(1, 6)
    inertia = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(data)
        inertia.append(model.inertia_)

    # Plot ks vs inertia
    plt.plot(ks, inertia, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data)
    iris_df.columns = iris.feature_names
    print(iris_df.head())
    optimal_clusters(iris_df)
    num_clust = int(input("What is the optimal number of clusters?"))
    model = KMeans(n_clusters=num_clust)
    model.fit(iris_df)
    labels = model.predict(iris_df)
    iris_df['labels'] = labels

    # Deriving personas from clusters
    for i in range(num_clust):
        clust = iris_df[iris_df['labels'] == i]
        clust = clust.drop(['labels'], axis=1)
        print('\nCluster', i)
        print(clust.head())





