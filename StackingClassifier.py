
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np

class StackingClassifier:
    def __init__(self,firstLevel,finalClf,metaFeatures,cv_for_fit=4):
        self.firstLevel=firstLevel
        self.finalClf=finalClf
        self.metaFeatures=metaFeatures
        self.cv_for_fit=cv_for_fit

    def fit(self,X,Y):
        X=np.array(X)
        Y=np.array(Y)
        for feat in self.metaFeatures:
            feat.fit(X)

        secondLevelInput=np.concatenate([feat.transform(X) for feat in self.metaFeatures],axis=1)

        folds=KFold(X.shape[0],n_folds=self.cv_for_fit,shuffle=True)
        firstLevelPredictions=[]
        for clf in self.firstLevel:
            miniPreds=[]
            for train,test in folds:
                xTrain=X[train,:]
                yTrain=Y[train]
                clf.fit(xTrain,yTrain)
                miniPreds.append(clf.predict_proba(X[test,:]))
            firstLevelPredictions.append(np.concatenate(miniPreds))

            # Do the final fit for each classifier
            clf.fit(X,Y)

        secondLevelInput=np.hstack([secondLevelInput]+firstLevelPredictions)

        


        shuffledY=Y[[b for a in folds for b in a[1]]]

        self.finalClf.fit(secondLevelInput,shuffledY)

    def predict(self,X):
        secondLevelInput=np.concatenate([feat.transform(X) for feat in self.metaFeatures]+
                                        [clf.predict_proba(X) for clf in self.firstLevel],axis=1)
        return self.finalClf.predict(secondLevelInput)



