from sklearn.cross_validation import cross_val_predict

class weightedRegressor:
    def __init__(self,regList,cv=3, weights='uniform'):
        self.regList=regList
        self.cv=cv

    def fit(self,X,Y):
        self.predictions=[]
        n=len(Y)
        self.scores=[]
        for reg in self.regList:
            self.predictions.append(cross_val_predict(reg,X,Y,cv=self.cv))
            score=sum([(p-a)**2. for (p,a) in zip(self.predictions,Y)])/n
            scores.append(score)

            # If working in Pandas, put in index
            reg.fit(X,Y)
            if weights=='uniform':
                self.weights=[1./len(self.regList)]*len(self.regList)
            elif weights=='distance':
                tot=sum(self.scores)
                self.weights=[s/tot for s in self.scores]


    def predict(X):
        return np.array(np.dot(self.weights,[reg.predict(X) for reg in self.regList]))

    


