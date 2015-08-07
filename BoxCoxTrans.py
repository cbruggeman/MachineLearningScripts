from scipy.stats import boxcox
from scipy.stats import boxcox_normmax

class BoxCoxTrans:
    def __init__(self,lmbda=None):
        self.lmbda=lmbda
        if lmbda==None:
            self.lmbdaGiven=True

    def fit(self,X):
        if not self.lmbdaGiven:
            self.lmbda=boxcox_normmax(X,method='mle')

    def fit_transform(self,X):
        if self.lmbdaGiven:
            transformed,self.lmbda=boxcox(X)
            return transformed
        else:
            return boxcox(X,self.lmbda)

    def transform(X):
        return boxcox(X,self.lmbda)