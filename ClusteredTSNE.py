from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import math

class ClusteredTSNE:
    def init(self,n_cluster=1000):
        self.n_cluster=1000

    def fit(self,X):
        self.clusterer=MiniBatchKMeans(n_cluster)
        self.clusterer.fit(trainX)
        self.clustCenters=clusterer.cluster_centers_
        self.tsn=TSNE()
        self.tsneProjections=tsn.fit_transform(clustCenters)

    def transform(self,X):
        clustDf=pd.DataFrame({'cluster':self.clusterer.predict(X)},index=trainX.index)
        tsneTransformedDf=clustDf.merge(pd.DataFrame(tsneProjections),right_index=True,left_on="cluster").iloc[:,1:]
        return tsneProjections

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

