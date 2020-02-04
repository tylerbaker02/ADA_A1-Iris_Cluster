
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


# Data Centering
def visualize_distributions_for_skew(data):
    for col in data.columns:
        print(col)
        sns_plot = sns.distplot(data.loc[:,col], bins=20)
        fig = sns_plot.get_figure()
        fig.savefig('batch_figures/' + col + ".png")
        fig.clear()

        frequency_log = np.log(data.lo[:, col] + 1)
        log_plot = sns.distplot(frequency_log)
        log_fig = log_plot.get_figure()
        log_fig.savefig('batch_figures/log' + col + ".png")
        plt.show()

def centering(data):
    return data.apply(np.log)


# Standardization
def standardization(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data))


# Decorrelation
def decorrelation(data):
    pca = PCA()
    pca.fit(data)
    return pd.DataFrame(pca.transform(data))


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
    return pd.DataFrame(pca_features)


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
    # print(iris_df.head())

    optimal_clusters(iris_df)
    num_clust = int(input("What is the optimal number of clusters?"))
    model = KMeans(n_clusters=num_clust)
    bdat = feature_reduction(standardization(decorrelation(standardization(centering(iris_df)))))

    model.fit(bdat)
    iris_df['labels'] = model.predict(bdat)
    iris_df['actual'] = iris.target_names[iris.target]
    ct = pd.crosstab(iris_df['labels'], iris_df['actual'])

    # Deriving personas from clusters
    for i in range(num_clust):
        clust = iris_df[iris_df['labels'] == i]
        clust = clust.drop(['labels', 'actual'], axis=1)
        print('\nCluster', i)
        print(clust.head())

    print(ct)





