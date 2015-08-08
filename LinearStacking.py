from scipy.optimize import minimize
from __future__ import division
from sklearn.cross_validation import cross_val_predict
from sklearn.covariance import empirical_covariance
from sklearn.linear_model import LinearRegression
import numpy as np

class weightedRegressor:
    def __init__(self,regList,
                 cv=3,
                 weighting='uniform',
                 biasWeighting=1.0,
                 stacker=LinearRegression(fit_intercept=False)):
        self.regList=regList
        self.cv=cv
        self.weighting=weighting
        self.biasWeighting=biasWeighting
        self.stacker=stacker

    def fit(self,X,Y):
        self.predictions=[]
        n=len(Y)
        self.MSEs=[]
        self.weights=[]
        for reg in self.regList:
            self.predictions.append(cross_val_predict(reg,X,Y,cv=self.cv))
            MSE=sum([(p-a)**2. for (p,a) in zip(self.predictions[-1],Y)])/n
            self.MSEs.append(MSE)
            reg.fit(X,Y)

        if self.weighting=='uniform':
            self.weights=[1./len(self.regList)]*len(self.regList)
        elif self.weighting=='score':
            tot=sum([1./s for s in self.MSEs])
            self.weights=[1./(s*tot) for s in self.MSEs]
        elif self.weighting=='varMin':
            self.covariance=empirical_covariance(np.array(self.predictions).T)
            self.weights=smallestVarianceWeights(self.covariance,self.MSEs,self.biasWeighting)
        elif self.weighting=='linearReg':
            self.stacker.fit(np.array(self.predictions).T,Y)
            self.weights=self.stacker.coef_
        
        print self.weights




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
                    bounds=bounds,
                    constraints=cons,
                    method='SLSQP').x
