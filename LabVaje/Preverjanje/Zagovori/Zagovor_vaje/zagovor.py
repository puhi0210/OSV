from types import DynamicClassAttribute
import numpy as np
from matplotlib import pyplot as plt
from Vaja_2.skripta import loadImage
from Vaja_4.skripta import displayImage3D as displayImage
from Vaja_7.skripta import changeSpatialDomain

def distancePoint2Line ( iL , iP ) :
    # y = k*x + n
    # kx - y + n = 0
    a = iL[0] #iL vhodna premica
    b = -1
    c = iL[1]
    y,x = iP
    oD = (abs(a*iP[0] + b*iP[1] +c)/(np.sqrt(a**2 + b**2))) #glej navodila, obrnjena X in Y
    return oD

def weightedGaussianFilter ( iS , iWR , iStdR , iW ) :
    #vaja 7
    #N, M = iS
    #sigma je presečišče na premici
    w1, w2 = iWR
    sgm1, sgm2 = iStdR

    kk = (sgm2 - sgm1) / (w2 - w1)
    nn = sgm1 - (kk*w1)

    oStd = kk * iW + nn

    M, N = iS #dimneziji filtra
    oK = np.zeros((M,N)) #kontra #PRI NP ZEROS VEDNO (()) DOKUMENTACIJA!
    m = int((M-1)/2) 
    n = int((N-1)/2)

    #lahko z uporabo enumerate
    for j, y in enumerate(np.arange(-n, n+1)):
        for i, x in enumerate(np.arange(-m, m+1)):
            oK[j, i] = 1/(2*np.pi*oStd**2)*np.exp(-(x**2+y**2)/(2*oStd**2)) #formula
    #brez enumerate
    for y in (np.arange(-n, n+1)):
        for x in (np.arange(-m, m+1)):
            oK[y + n, x + m] = 1/(2*np.pi*oStd**2)*np.exp(-(x**2+y**2)/(2*oStd**2)) #formula

    oK = oK/oK.sum() #NORMIRAMO, DRUGAČE SLIKI NABIJE SVETLOST
    return oK , oStd #oK je jedro filtra, oStd pa standardni odklon

def imitateMiniature ( iImage , iS , iStdR , iL , iD0 ) :
    #sliko moramo ekstrapolirati! Zato da filter lahko gre tudi čez robove

    Y, X = iImage.shape
    d1 = distancePoint2Line(iL, [0,0])
    d2 = distancePoint2Line(iL, [0,Y-1])
    d3 = distancePoint2Line(iL, [X-1,Y-1])
    d4 = distancePoint2Line(iL, [X-1, 0])

    d_max = max(d1,d2,d3,d4) #najbolj oddaljen vogal

    M, N = iS #dimneziji filtra
    m = int((M-1)/2) 
    n = int((N-1)/2)

    iImage = changeSpatialDomain('enlarge', iImage, iX = m, iY = n) #mode = 'extrapolate'

    Y, X = iImage.shape #velikost slike se je spremenila

    oImage = np.array(iImage, dtype = float)
    oVal = []

    for y in range (n, Y-n):
        for x in range(m, X-m):
            d = distancePoint2Line(iL, [x,y]) #razdalja od premice
            if d > iD0:
                K, std = weightedGaussianFilter(iS, [iD0, d_max], iStdR, d)
                patch = iImage[y-n:y+n+1, x-m:x+m+1]
                oImage[y,x] = np.sum(patch*K)
                oVal.append([d, std]) #v seznam dodajamo razdaljo in std za vsako točko
    oImage = changeSpatialDomain('reduce', oImage, iX = m, iY = n)

    return oImage , oVal

if __name__ == "__main__":
    I = loadImage(r"Vaja_Zagovor\train-400x240-08bit.raw", (400,240), np.uint8) #r pred "" če pride do napake zaradi /t
    #PNG IN JPEG SLIKE SE NALAGA DRUGAČE Z IMSHOW, POGLEJ VAJO 01
    displayImage(I)
    iP = [0,100]
    iL = [0.22, 100]
    test = distancePoint2Line(iL, iP) #ko izberemo točko na premici bi mogli dobiti razdaljo 0

    #2. naloga
    wr = [0,10]
    stdr = [0.1, 10]
    sz = [7,7]
    K, std = weightedGaussianFilter(sz, wr, stdr, 10)

    #3. naloga
    mI, vals = imitateMiniature(I, sz, iStdR = stdr, iL = iL, iD0 = 25)
    displayImage(mI)

    vals = np.array(vals)

    plt.figure()
    plt.plot(vals[:,0], vals[:,1])
    plt.grid()
    plt.show()