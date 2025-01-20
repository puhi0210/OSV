import numpy as np
import matplotlib.pyplot as plt


def distancePoint2Line(iL , iP):
    a = iL[0]
    b = -1
    c = iL[1]

    oD = abs(a*iP[0] + b*iP[1] + c) / np.sqrt(a**2 + b**2)


    return oD

def weightedGaussianFilter(iS , iWR , iStdR , iW):
    oK = np.zeros((iS[1] , iS[0]))
    


    return oK , oStd