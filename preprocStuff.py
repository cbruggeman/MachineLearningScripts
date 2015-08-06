# Blur By Correlation

# Want to find the correlation of everything, and then blur cells together
# according to how correlated they are.

class BlurByCor:

    def __init__ (self,corType='pearson',corCutoff=0,sdCutoff=0):
        self.corType=corType
        self.corCutoff=corCutoff
        self.sdCutoff=sdCutoff

    def fit(self,data):
        self.cols=data.columns[data.std(axis=0)>self.sdCutoff]
        self.corMat=data.loc[:,self.cols].corr(method=self.corType)
        self.corMat[self.corMat<self.corCutoff]=0

    def transform(self,data):
        return pd.DataFrame(np.dot(data.loc[:,self.cols].as_matrix(),self.corMat),
                            index=data.index,
                            columns=self.cols)

    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)





# Want to add clusters to the feature set

def addUnsupResults(X,listOfLearners,numCompList=range(2,11)):
    
    # Need to Find a Way to Make Col Names Unique after
    newX=X.copy()
    for learner in listOfLearners:
        for numComp in numCompList:
            newX=pd.merge(newX,
                       pd.DataFrame(learner(numComp).fit_transform(X),index=X.index),
                       left_index=True,
                       right_index=True)
    return newX