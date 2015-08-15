import numpy as np

class DropColumns:
    def __init__(self,columns):
        self.columns=columns

    def fit(self,X=None,Y=None):
        pass

    def transform(self,X=None,Y=None):
        return np.array(X)[:,np.array(self.columns)]
    
    def fit_transform(self,X=None,Y=None):
        return self.transform(X)