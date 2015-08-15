class ExtraClassClassifier:
    def __init__(self,baseEstimator=LDA(),numClust=60):
        self.baseEstimator=baseEstimator
        self.numClust=numClust

    def fit(self,X,Y):
        X=pd.DataFrame(X)
        Y=pd.Series(Y)
        self.classes_=np.unique(Y).astype(str)
        seriesList=[]
        for i in np.unique(Y):
            temp=X[Y==i]
            self.clust=MiniBatchKMeans(self.numClust)
            seriesList.append(pd.Series(self.clust.fit_predict(temp),index=temp.index))
        newY=np.array(pd.Series(Y).astype(str)+pd.concat(seriesList).astype(str))
        X=np.array(X)
        self.baseEstimator.fit(X,newY)
        self.extraClasses_=self.baseEstimator.classes_

    def predict(self,X,Y=None):
        return np.apply_along_axis(lambda x: x[0],axis=0, arr=self.baseEstimator.predict(X))


    def predict_proba(self,X,Y=None):
        extraClassPredictedProba=self.baseEstimator.predict_proba(X)
        return np.array([sum([extraClassPredictedProba[:,c] for c in range(len(self.extraClasses_)) if self.extraClasses_[c][0]==realClass[0]]) for realClass in self.classes_]).T

