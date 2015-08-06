class RandomEstimator():
	def __init__(self,listOfEstimators):
		self.listOfEstimators=listOfEstimators
		self.est=None

	def fit(self,X,Y,weights=None):
		self.weights=weights
		if selt.est==None:
			self.est=random.choice(self.listOfEstimators)
		self.est.fit(X,Y,weights)

	def fit_transform(self,X,Y):
		self.fit(X,Y)
		return self.transform(X,Y)

	def transform(self,X,Y):
		return self.est.transform(X,Y)

	def predict_proba(X):
		return self.est.predict_proba(X)

	def predict(self,X):
		return self.est.predict(X)



class groupedConditionedLearner():
    def __init__(self,learner,cluster):
        self.learner=learner
        self.cluster=cluster
    
    
    def fit(self,X,Y):
        X=np.array(X)
        Y=np.array(Y)
        clusterArray=self.cluster.predict(X)
        self.clusterNames=np.unique(clusterArray)
        self.clusterLearners={}
        for name in self.clusterNames:
            learner=clone(self.learner)
            clusterIndex=np.where(clusterArray==name)
            cX=X[clusterIndex]
            cY=Y[clusterIndex]
            learner.fit(cX,cY)
            self.clusterLearners[name]=learner
        
    
    def predict_proba_(self,X):
        X=np.array(X)
        clusters=self.cluster.predict(X)
        listAnswer=[self.clusterLearners[clusters[c]].predict_proba_(X[c])[0] for c in range(len(X))]
        return np.array(listAnswer)
    
    def predict(self,X):
        X=np.array(X)
        clusters=self.cluster.predict(X)
        listAnswer=[self.clusterLearners[clusters[c]].predict(X[c])[0] for c in range(len(X))]
        return np.array(listAnswer)


    def score(self,X,Y):
    	preds=self.predict(X)
        Y=np.array(Y)
    	return sum([preds[c]==Y[c] for c in range(len(preds))])/float(len(preds))