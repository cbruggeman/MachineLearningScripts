from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE

class ClusteredTSNE:
    def __init__(self,dimension=2,n_cluster=1000):
        self.n_cluster=n_cluster
        self.dimension=dimension

    def fit(self,X):
        self.clusterer=MiniBatchKMeans(self.n_cluster)
        self.clusterer.fit(X)
        self.clustCenters=self.clusterer.cluster_centers_
        self.tsn=TSNE(self.dimension)
        self.tsneProjections=self.tsn.fit_transform(self.clustCenters)

    def transform(self,X):
        clustDf=pd.DataFrame({'cluster':self.clusterer.predict(X)})
        tsneTransformedDf=clustDf.merge(pd.DataFrame(self.tsneProjections),right_index=True,left_on="cluster").iloc[:,1:]
        return np.array(tsneTransformedDf)

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)