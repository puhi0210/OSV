from pickletools import uint8
from matplotlib import pyplot as plt
import numpy as np
from Vaja_2.skripta import loadImage
from Vaja_4.skripta import displayImage3D as displayImage

## IZ 7. VAJE ZA 2. NALOGO

def changeSpatialDomain ( iType , iImage , iX , iY , iMode = None , iBgr =0) :
    Y,X = iImage.shape
    #iX število stoplcev ki jih dodamo, iY število vrstic
    if iType == 'enlarge':
        if iMode is None:
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage #glej sliko
        #elif doma pomoč z np.vstack in np.hstack
        elif iMode == "constant":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) + iBgr
            oImage[iY:Y + iY, iX:X + iX] = iImage
        elif iMode == "extrapolation":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage  #input slika gre na sredino nove slike

            oImage[:iY, iX:iX + X] = iImage[0, :] #SEVER
            oImage[iY + Y:, iX:iX + X] = iImage[-1, :] #JUG

            oImage[:, :iX] = np.tile(oImage[:, iX:iX + 1], (1, iX))  #VZHOD
            oImage[:, iX + X:] = np.tile(oImage[:, iX + X - 1:iX + X], (1, iX))#ZAHOD
        elif iMode == "reflection":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage 

            # Za zgornji rob
            for i in range(iY):
                if not i // Y % 2:
                    oImage[iY - i, :] = oImage[iY + i % Y, :]
                else:
                    oImage[iY - i, :] = oImage[iY + Y - i % Y, :]

            # Za levi rob
            for i in range(iX):
                if not i // X % 2:
                    oImage[:, iX - i] = oImage[:, iX + i % X]
                else:
                    oImage[:, iX - i] = oImage[:, iX + X - i % X]

            # Za desni rob
            for i in range(iX):
                oImage[:, iX + X + i] = oImage[:, iX + X -1 -i]
            
            # Za spodnji rob
            for i in range(iY):
                oImage[iY + Y + i, :] = oImage[iY + Y -1 -i, :]
        elif iMode == "period":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage 

            # Za zgornji rob
            for i in range(iY):
                oImage[iY - i, :] = oImage[iY + Y - i % Y, :]

            # Za spodnji rob
            for i in range(iY + Y, 2 * iY + Y):
                oImage[i, :] = oImage[i - Y, :]

            # Za spodnji rob
            for i in range(iX):
                oImage[:, iX - i] = oImage[:, iX + X - i % X]
            
            # Za desni rob
            for i in range(iX + X, 2 * iX + X):
                oImage[:, i] = oImage[:, i - X]
        else:
            raise ValueError("napačen iMode")
    elif iType == 'reduce':
        oImage = iImage[iY:Y - iY, iX:X - iX] #lih obratno
    else:
        raise ValueError("napačen iType")
        
    return oImage

def spatialFiltering ( iType , iImage , iFilter , iStatFunc = None, iMorphOp = None ) :
    M, N = iFilter.shape #dimneziji filtra
    m = int((M-1)/2) #koliko praznih vrstic/stolpcev imamo
    n = int((N-1)/2)

    iImage = changeSpatialDomain('enlarge', iImage, iX = n, iY = m)

    Y, X = iImage.shape #dobimo dimnezijo slike
    oImage = np.zeros((Y,X), dtype = float) #inicializiramo oImage, velikosti slike, same ničle

    for y in range(n, Y-n): #do Y-n, ker gre do konca slike minus polovica filtra
        for x in range(m, X-m):
            patch = iImage[y-n:y+n+1, x-m:x+m+1] #tole je filter v sliki, kje se nahaja. +1 je notri ker python z : gleda do (ne vključno)

            if iType == 'kernel':
                oImage[y, x] = np.sum(patch * iFilter) #filter na sliki množimo z parametri filtra, TRANSPONIRANJE ZA MNOŽENJE Z MATRIKAMI KI NISO NxN
            elif iType == 'kernel_mismatch':
                oImage[y, x] = np.sum(patch * iFilter.T)
            elif iType == 'statistical':
                oImage[y,x] = iStatFunc(patch) #iStatFunc gre notr np.mean ipd, poglej prejšne vaje
            elif iType == 'morphological':
                R = patch[iFilter != 0] #pogleda kje v matriki so vrednosti 1 in filtrira
                if iMorphOp == 'erosion':
                    oImage[y,x] = R.min() #np.min(R)
                elif iMorphOp == 'dilation':
                    oImage[y,x] = R.max()
                else:
                    raise NotImplementedError('Napačen iMorphOp')
            else:
                raise NotImplementedError('Napačen iType')

    oImage = changeSpatialDomain('reduce', oImage, iX = m, iY = n)
    return oImage


#2. Naloga : Weighed average filter
def weightedAverageFilter(iM, iN, iValue):
    filter_center = ((iM - 1) // 2, (iN - 1) // 2) #sredina filtra, glej slikco iz vaj
    oFilter = np.zeros([iM, iN]) #inicializiramo filter

    for i in range(iM):
        for j in range(iN):
            center = np.sqrt((i - filter_center[0])**2 + (j - filter_center[1])**2) #evklidska razdalja
            top = np.ceil(center)
            oFilter[i, j] = iValue ** (max(iM, iN) // 2 + 2 - top)
            #na robovih vedno enke
            oFilter[0,0] = 1
            oFilter[iM-1, 0] = 1
            oFilter[iM-1, iN-1] = 1
            oFilter[0, iN - 1] = 1
    oFilter = oFilter/np.sum(oFilter)
    return oFilter

## TU SE ZAČNE ZAGOVOR

I = plt.imread("Zagovori/ZagovorBL/bled-lake-decimated-uint8.jpeg")

# 1. Naloga
def get_blue_region(iImage, iThreshold):
    #gledamo blue kanal BLUE BLUE BLUE (rgB)
    # o jao modeli so dal B kanal v [1]
    Y,X,C = iImage.shape
    B = iImage[:,:,1]
    R = iImage[:,:,0]
    G = iImage[:,:,2]
    
    oImage = np.zeros((Y,X))
    maska_RGB = np.zeros((Y,X))

    for y in range(Y):
        for x in range(X):
            if B[y][x] > iThreshold:
                #oImage[y][x] = B[y][x] #zapišemo vrednost tam v matriko
                oImage[y][x] = 1  #NAREDI SUPER MASKO, NA KONCU MNOŽI Z 255
            else:
                oImage[y][x] = 0

    for y in range(Y):
        for x in range(X):
            if (B[y][x] > iThreshold and G[y][x] > iThreshold and R[y][x] > iThreshold):
                maska_RGB[y][x] = 255  #zapišemo vrednost tam v matriko
                #oImage[y][x] = 1  NAREDI SUPER MASKO, NA KONCU MNOŽI Z 255
            else:
                maska_RGB[y][x] = 0

    maska = np.zeros((Y,X))
    maska = np.where(oImage > iThreshold, 1, 0)

    maska_2 = np.zeros((Y,X))
    maska_2 = np.where((B > iThreshold) & (R > iThreshold) & (G > iThreshold), 1, 0)

    maska_full = np.zeros((Y,X))
    maska_full = np.where((maska & maska_2 ), 1, 0)
    maska_full = maska_full* 255

    #oImage = oImage * 255
    #print(maska_RGB)
    #displayImage(maska_full)
    oImage *= 255
    return oImage

# 3. NALOGA
def find_edge_coordinates(iImage):
    #ideja je filtirati z sobelom in dobiti ven koordinate kjer je belo
    Sb_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    Sb_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    sobel_x = spatialFiltering('kernel', iImage, Sb_x)
    sobel_y = spatialFiltering('kernel', iImage, Sb_y)    

    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    #normaliziramo sobela
    sobel -= np.min(sobel)
    sobel /= np.max(sobel)
    sobel *= 255

    #print(sobel)
    Y,X = iImage.shape

    oEdges = []

    for y in range(Y):
        for x in range(X):
            if sobel[y][x] == 255:
                oEdges.append((x,y))

    return oEdges, sobel

# 4. NALOGA
def compute_distances(iImage, iMask = None):

    return oImage


if __name__ == "__main__":
    print(np.shape(I))

    # 1. Naloga
    # za dejansko masko jezera je najbolje uporabiti treshold 210
    # v navodilih kr neki
    blue = get_blue_region(I, 237)
    displayImage(blue, "modro jezero")


    maska = get_blue_region(I, 210)
    displayImage(maska, "Slika z pragom 210")

    
    # 2. Naloga
    filter = np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0]
    ])

    lake_morf = spatialFiltering('morphological', maska, filter, _, 'dilation')
    #displayImage(lake_morf, "Slika po filtriranju morf" )

    lake_dil = spatialFiltering('morphological', lake_morf, filter, _, 'erosion')

    #displayImage(lake_dil, "Slika po filtriranju z dil")

    # 3. Naloga
    edges, sobel = find_edge_coordinates(lake_dil)
    print(edges)

    #4. naloga
    