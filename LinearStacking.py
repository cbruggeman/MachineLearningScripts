from scipy.optimize import minimize
from __future__ import division
from sklearn.cross_validation import cross_val_predict
from sklearn.covariance import empirical_covariance
import numpy as np

class weightedRegressor:
    def __init__(self,regList,cv=3, weighting='uniform', biasWeighting=1.0):
        self.regList=regList
        self.cv=cv
        self.weighting=weighting
        self.biasWeighting=biasWeighting

    def fit(self,X,Y):
        self.predictions=[]
        n=len(Y)
        self.scores=[]
        self.weights=[]
        for reg in self.regList:
            self.predictions.append(cross_val_predict(reg,X,Y,cv=self.cv))
            MSE=sum([(p-a)**2. for (p,a) in zip(self.predictions[-1],Y)])/n
            self.MSEs.append(mse)
            reg.fit(X,Y)

        if self.weighting=='uniform':
            self.weights=[1./len(self.regList)]*len(self.regList)
        elif self.weighting=='score':
            tot=sum([1./s for s in self.scores])
            self.weights=[1./(s*tot) for s in self.scores]
        elif self.weighting=='varMin':
            self.covariance=empirical_covariance(self.predictions)
            self.weights=smallestVarianceWeights(self.covariance,self.MSEs)




    def predict(self,X):
        # If working in Pandas, put in index
        return np.array(np.dot(self.weights,[reg.predict(X) for reg in self.regList]))

    def score(self,X,Y):
        preds=self.predict(X)
        yMean=float(sum(Y))/len(Y)
        SSTotal=sum([(y-yMean)**2 for y in Y])
        SSRegression=sum([(p-a)**2 for p,a in zip(preds,Y)])
        return (1-SSRegression/SSTotal)



def smallestVarianceWeights(covariance,biases,biasWeighting=0.):
    dimension=covariance.shape[0]
    biases=np.array(biases)
    # Define the objective function
    func=lambda x: covariance.dot(x).dot(x)+biasWeighting*np.dot(x,biases)
    
    x0=np.ones(dimension)/dimension
    
    grad=lambda x: 2*covariance.dot(x)+biasWeighting*biases
    
    cons=({'type':'eq',
          'fun':lambda x: sum(x)-1,
          'jac':lambda x: np.ones(dimension)})
    
    bounds=[(0,None) for i in range(dimension)]
    
    return minimize(fun=func,
                    jac=grad,
                    x0=x0,
                    constraints=cons,
                    method='SLSQP',
                    options={'disp': True})
