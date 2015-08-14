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


class stdFeature:
    def __init__(self):
        pass

    def fit(self,X):
        pass

    def transform(self,X):
        return X.std(axis=1)

class directionalStdFeature:
    def __init__(self,axis,dimensions=None):
        self.axis=axis
        self.dimensions=dimensions

    def fit(self,X):
        # If nothing given, assume square image
        if self.dimensions==None:
            self.dimensions=(int(X.shape[1]**0.5))*2

    def transform(self,X):
        def std(x):
            im=x.reshape(self.dimensions)
            total=0
            weightedTotal=0
            num=self.dimensions[self.axis]
            for i in range(num):
                s=im.take(i,axis=self.axis).sum()
                total+=s
                weightedTotal+=i*s
            mean=float(weightedTotal)/total
            RSE=0
            for i in range(num):
                s=im.take(i,axis=self.axis).sum()
                RSE+=(mean-i)**2.*s
            return (RSE/total)**(0.5)

        return np.apply_along_axis(std,axis=1,arr=X)



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

def largestEigenvector(matrix,angle=False):
    values,vectors=np.linalg.eig(matrix)
    x,y=vectors[np.argmax(values)]
    if angle:
        return math.atan(y/x)
    else return np.array(x,y)