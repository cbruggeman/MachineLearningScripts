from scipy import ndimage
import numpy as np
import math

class sumFeature:
    def __init__(self):
        pass

    def fit(self,X):
        pass

    def transform(self,X):
        return X.sum(axis=1)

    def fit_transform(self,X):
        return self.transform(X)


class stdFeature:
    def __init__(self):
        pass

    def fit(self,X):
        pass

    def transform(self,X):
        return X.std(axis=1)

    def fit_transform(self,X):
        return self.transform(X)

class directionalStdFeature:
    def __init__(self,axis,dimensions=None):
        self.axis=axis
        self.dimensions=dimensions

    def fit(self,X):
        # If nothing given, assume square image
        if self.dimensions==None:
            self.dimensions=[int(X.shape[1]**0.5)]*2

    def transform(self,X):
        X=np.array(X)
        return np.apply_along_axis(lambda x: (varByAxis(x.reshape(self.dimensions),
                                                        axis=self.axis)**0.5),
                                    axis=1,
                                    arr=X)

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)



def meanByAxis(im,axis):
    total=0
    weightedTotal=0
    num=im.shape[axis]
    for i in range(num):
            s=im.take(i,axis=axis).sum()
            total+=s
            weightedTotal+=i*s
    return (float(weightedTotal)/total)

def varByAxis(im,axis):
    mean=meanByAxis(im,axis)
    RSE=0
    total=0
    num=im.shape[axis]
    for i in range(num):
        s=im.take(i,axis=axis).sum()
        total+=s
        RSE+=(mean-i)**2.*s
    return (RSE/total)**(0.5)

def coVar(im):
    meanY=meanByAxis(im,0)
    meanX=meanByAxis(im,1)
    varY=varByAxis(im,0)
    varX=varByAxis(im,1)
    total=0
    cross=0

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            total+=im[i,j]
            cross+=(i-meanY)*(j-meanX)*im[i,j]

    return (float(cross)/(total*(varX*varY)**(0.5)))

def imageCovarianceMatrix(im):
    varX=varByAxis(im,1)
    varY=varByAxis(im,0)
    covarXY=coVar(im)
    return np.array([[varX,covarXY],[covarXY,varY]])

def eigenvectorAngles(matrix):
    values,vectors=np.linalg.eig(matrix)
    x1,y1=vectors[np.argmax(values)]
    x2,y2=vectors[np.argmin(values)]
    return np.array([math.atan(y1/x1),math.atan(y2/x2)])

class imageEigenAxes:
    def __init__(self,dimensions=None, eigenAngles=True, eigenRatio=True, eigenValues=True):
        self.dimensions=dimensions
        self.eigenAngles=eigenAngles
        self.eigenRatio=eigenRatio

    def fit(self,X):
        # If nothing given, assume square image
        if self.dimensions==None:
            self.dimensions=[int(X.shape[1]**0.5)]*2

    def transform(self,X):
        X=np.array(X)
        return np.apply_along_axis(lambda x: (eigenvectorAngles(imageCovarianceMatrix(x.reshape(self.dimensions)))),
                            axis=1,
                            arr=X)

