

class ExtraClassClassifier:
	def __init__(self,baseEstimator=LDA(),numClust=60):
		self.baseEstimator=baseEstimator
		self.numClust=numClust

	def fit(self,X,Y):
		X=np.array(X)
		Y=np.array(Y)
		for i in np.unique(Y):
        	temp=X[Y==i]
	        clust=MKM(self.numClust)
	        seriesList.append(pd.Series(clust.fit_predict(temp),index=temp.index))
    newY=Y+pd.concat(seriesList).astype(str)
