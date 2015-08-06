from scipy.optimize import minimize
from __future__ import division
import numpy as np
import operator



def smallVarCombination(covariance):
    dimension=covariance.shape[0]
    
    # Define the objective function
    func=lambda x: covariance.dot(x).dot(x)
    
    x0=np.ones(dimension)/dimension
    
    grad=lambda x: 2*covariance.dot(x)
    
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

def quadraticWeightedVote(outputs):
    if type(outputs)==list:
        p=len(outputs)
        n=len(outputs[0])
    else:
        n,p=outputs.shape

    covar=np.empty((p,p))
    for i in range(p):
        covar[i,i]=1
        for j in range(i+1,p):
            covar[i,j]=sum([a==b for a,b in zip(outputs[i],outputs[j])])/n
            covar[j,i]=covar[i,j]

    weights=smallVarCombination(covar)
    print weights
    # Vectorize this later and check performance change
    vote=[]
    for i in range(n):
        votes={}
        for voter in range(p):
            try:
                votes[outputs[voter][i]]+=weights[voter]
            except KeyError:
                votes[outputs[voter][i]]=weights[voter]
        vote.append(max(votes.iteritems(), key=operator.itemgetter(1))[0])

    return vote




    