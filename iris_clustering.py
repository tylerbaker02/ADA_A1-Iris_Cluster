""" Cluster the Iris Dataset
Author: Tyler Baker
Class:  DAT-310-01
Certification of Authenticity:
I certify that this is entirely my own work, except where I have given fully documented
references to the work of others. I understand the definition and consequences of
plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
assessing this assignment reproduce this assignment and provide a copy to another member
of academic staff and / or communicate a copy of this assignment to a plagiarism checking
service(which may then retain a copy of this assignment on its database for the purpose
of future plagiarism checking).
"""
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
    """ Create Graphs displaying distributions before and after centering the data."""
    for col in data.columns:
        sns_plot = sns.distplot(data.loc[:, col], bins=20)
        fig = sns_plot.get_figure()
        fig.savefig('batch_figures/' + col + ".png")
        fig.clear()

        frequency_log = np.log(data.loc[:, col] + 1)
        log_plot = sns.distplot(frequency_log)
        log_fig = log_plot.get_figure()
        log_fig.savefig('batch_figures/log' + col + ".png")
        plt.show()


def centering(data):
    """ Center the data using logarithms. """
    return data.apply(np.log)


# Standardization
def standardization(data):
    """ Standardize the data using StandardScalar."""
    scaler = StandardScaler()
    scaler.fit(data)
    return pd.DataFrame(scaler.transform(data))


# Decorrelation
def correlation_map(data):
    """ Display the correlation of the data on a heatmap."""
    sns.heatmap(data.corr(), annot=True)


def decorrelation(data):
    """ Decorrelate the data using PCA. """
    pca = PCA()
    pca.fit(data)
    return pd.DataFrame(pca.transform(data))


# Feature Reduction
def feature_variance_check(data):
    """ Check the variance for each PCA feature of the data. """
    pca = PCA()
    pca.fit(data)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()


def feature_reduction(data, num_com=2):
    """ Reduce the features to the number given, default of 2. """
    pca = PCA(n_components=num_com)
    pca.fit(data)
    pca_features = pca.transform(data)
    print(pca_features.shape)
    return pd.DataFrame(pca_features)


# Finding the optimal number of clusters
def optimal_clusters(data):
    """ Build an elbow graph for the given data. """
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

    # Load iris dataset and get feature names
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data)
    iris_df.columns = iris.feature_names
    # Check for if certain functions should be used or not
    # visualize_distributions_for_skew(iris_df)
    # correlation_map(iris_df)
    # feature_variance_check(iris_df)
    optimal_clusters(iris_df)

    # Ask the user how many clusters should be used
    num_clust = int(input("What is the optimal number of clusters?"))

    # Use cleaning functions on data
    bdat = standardization(decorrelation(standardization(centering(iris_df))))

    # create kmeans model, predict the clusters, label original dataframe with clusters
    model = KMeans(n_clusters=num_clust)
    model.fit(bdat)
    iris_df['labels'] = model.predict(bdat)
    iris_df['actual'] = iris.target_names[iris.target]

    # build a crosstab using the labels and actual values from original dataframe
    ct = pd.crosstab(iris_df['labels'], iris_df['actual'])

    # Deriving personas from clusters
    # Give first 5 values of each cluster
    for i in range(num_clust):
        clust = iris_df[iris_df['labels'] == i]
        clust = clust.drop(['labels', 'actual'], axis=1)
        print('\nCluster', i)
        print(clust.head())

    # print the crosstab that was built earlier
    print(ct)





