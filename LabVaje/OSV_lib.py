
#   Knjižnica funkcij za laboratorijske vaje pri predmetu OSV


import matplotlib.pyplot as plt
import numpy as np

# Funkcija za nalaganje slike iz datoteke
def loadImage(iPath, iSize, iType):
    fid = open(iPath, 'rb') # Odpremo datoteko v binarnem načinu
    buffer = fid.read()     # Preberemo vsebino datoteke v buffer
    buffer_len = len(np.frombuffer(buffer=buffer, dtype=iType)) # Dolžina bufferja, pretvorjena v polje tipa iType
    if buffer_len != np.prod(iSize):    # Preverimo, ali dolžina prebranih podatkov ustreza specifikaciji velikosti
        raise ValueError('Size of the input data does not match the specified size')
    else:   # Oblikovanje izhodne slike glede na podane dimenzije
        oImage_Shape = (iSize[1],iSize[0])
    # Pretvorimo buffer v numpy ndarray s specifičnimi dimenzijami in podatkovnim tipom
    oImage = np.ndarray(oImage_Shape, dtype = iType, buffer = buffer, order='F')
    return oImage   # Vrne prebrano sliko


# Funkcija za prikaz slike
def displayImage(iImage, iTitle = '', iGridX = None, iGridY = None):
    fig = plt.figure()  # Ustvarimo figuro za prikaz slike
    plt.title(iTitle)   # Ustvarimo figuro za prikaz slike

    if iGridX is not None and iGridY is not None:
        stepX = iGridX[1] - iGridX[0]
        stepY = iGridY[1] - iGridY[0]

        extent = (
            iGridX[0] - 0.5 * stepX,
            iGridX[-1] + 0.5 * stepX,
            iGridY[-1] + 0.5 * stepY,
            iGridY[0] - 0.5 * stepY
        )
    else:
        extent = (
            0 - 0.5, 
            iImage.shape[1] - 0.5,
            iImage.shape[0] - 0.5,
            0 - 0.5,
        )

    # Prikaz slike s sivinsko barvno karto in določenimi mejnimi vrednostmi
    plt.imshow(iImage, 
               cmap='gray',
               vmin=0,
               vmax=255,
               aspect='equal',
               extent=extent,
               )
    plt.show()
    return fig  # Vrne figuro


# Funkcija za shranjevanje slike v RAW formatu
def saveImage (iImage, iPath, iType):
    oImage = open(iPath, 'wb')  # Odpri datoteko za pisanje v binarnem načinu ('wb' - write binary)
    # Pretvori numpy array v niz bajtov z določenim podatkovnim tipom in fortranovskim redom ('F') in ga zapiši v datoteko
    oImage.write(iImage.astype(iType).tobytes(order = 'F'))
    oImage.close()  # Zapri datoteko


# Funkcija za izračun histograma, normaliziranega histograma in kumulativne porazdelitve verjetnosti sivinskih vrednosti slike
def computeHistogram(iImage):
    nBits = int(np.log2(iImage.max()))+1    # Izračunamo število bitov potrebnih za reprezentacijo nivojev sivin
    oLevels = np.arange(0, 2 ** nBits, 1)   # Ustvarimo vektor nivojev sivin
    iImage = iImage.astype(np.uint8)    # Pretvorimo sliko v 8-bitni format
    oHist = np.zeros(len(oLevels))      # Inicializiramo histogram z ničlami

    # Izračunamo histogram
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            oHist[iImage[y,x]] = oHist[iImage[y,x]] + 1

    oProb = oHist / iImage.size # Izračunamo normaliziran histogram
    oCDF = np.zeros_like(oHist) # Inicializiramo kumulativno porazdelitev z ničlami

    # Izračunamo kumulativno porazdelitev
    for i in range(len(oProb)):
        oCDF[i] = oProb[:i+1].sum()

    return oHist, oProb, oCDF, oLevels  # Vrne histogram, normaliziran histogram, kumulativno porazdelitev in nivoje sivin


# Funkcija za prikaz histograma
def displayHistogram(iHist, iLevels, iTitle):
    plt.figure()    # Ustvarimo novo figuro  
    plt.title(iTitle)   # Nastavimo naslov histograma
    # Prikaz histograma s podanimi nivoji in vrednostmi
    plt.bar(iLevels, iHist, width=1, edgecolor="darkred", color="red")
    plt.xlim((iLevels.min(), iLevels.max()))    # Nastavimo meje x-osi
    plt.ylim((0, 1.05 * iHist.max()))   # Nastavimo meje y-osi
    plt.show()


# Funkcija za za določanje slike z izravnanim histogramom
def equalizeHistogram(iImage):
    _, _, CDF, _ = computeHistogram(iImage) # Izračunamo kumulativno porazdelitev
    nBits = int(np.log2(iImage.max()))+1    # Izračunamo število bitov potrebnih za reprezentacijo nivojev sivin
    max_intensity = 2 ** nBits + 1  # Določimo maksimalno intenziteto
    oImage = np.zeros_like(iImage)  # Inicializiramo izhodno sliko

    # Izračunamo novo intenziteto za vsako slikovno točko
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            old_intensity = iImage[y,x]
            new_intensity = np.floor(CDF[old_intensity] * max_intensity)
            oImage[y,x] = new_intensity

    return oImage


# Funkcija za izračun entropije slike
def computeEntropy(iImage):
    _, imgProb, _, _ = computeHistogram(iImage) # Izračunamo verjetnostno porazdelitev
    oEntropy = 0

    # Izračunamo entropijo slike
    for i in range(len(imgProb)):
        if imgProb[i] != 0:
            oEntropy += imgProb[i]* np.log2(imgProb[i]) *(-1)

    return oEntropy # Vrne izračunano entropijo


# Funkcija za dodajanje Gaussovega šuma sliki
def addNoise(iImage, iStd): # iStd - standardna diviacija
    oNoise = np.random.randn(iImage.shape[0],iImage.shape[1])* iStd + iStd # Ustvarimo Gaussov šum
    oImage = iImage + (oNoise - iStd)   # Dodamo šum originalni sliki
    oImage = np.clip(oImage, 0, 255)    # Poskrbi, da vrednosti ostanejo v intervalih 0-255 

    return oImage, oNoise   # Vrne sliko z dodanim šumom in matriko šuma






# VAJA 4


def getPlanerCrossSection(iImage, iDim, iNormVec, iLoc):
    Y,X,Z = iImage.shape
    dx, dy, dz = iDim

    if iNormVec == [1,0,0]:
        oCS = iImage[:, iLoc, :].T
        oH = np.arange(Y) * dy
        oV = np.arange(Z) * dz
    elif iNormVec == [0,1,0]:
        oCS = iImage[iLoc, :, :].T
        oH = np.arange(X) * dx
        oV = np.arange(Z) * dz
    elif iNormVec == [0,0,1]:
        oCS = iImage[:, :, iLoc]
        oH = np.arange(X) * dx
        oV = np.arange(Y) * dy

    return np.array(oCS), oH, oV

def getPlanarProjection(iImage, iDim, iNormVec, iFunc):
    Y, X, Z = iImage.shape
    dx, dy, dz = iDim

    if iNormVec == [1,0,0]:
        oP = iFunc(iImage, axis = 1).T
        oH = np.arange(Y) * dy
        oV = np.arange(Z) * dz
    elif iNormVec == [0,1,0]:
        oP = iFunc(iImage, axis = 0).T
        oH = np.arange(X) * dx
        oV = np.arange(Z) * dz
    elif iNormVec == [0,0,1]:
        oP = iFunc(iImage, axis = 2)
        oH = np.arange(X) * dx
        oV = np.arange(Y) * dy

    return oP, oH, oV


# VAJA 5

def scaleImage(iImage, iA, iB):
    oImage = np.array(iImage, dtype=float)
    oImage = iImage * iA + iB
    return oImage


def windowImage(iImage , iC , iW):
    oImage = np.array(iImage, dtype=float)
    oImage = (255/iW) * (iImage -(iC -(iW/2)))

    oImage[iImage < iC - iW/2] = 0
    oImage[iImage > iC + iW/2] = 255

    return oImage


def sectionalScaleImage(iImage, iS, oS):
    oImage = np.array(iImage, dtype=float)
    
    for i in range(len(iS)-1):
        sL = iS[i]
        sH = iS[i+1]

        idx = np.logical_and(iImage >= sL, iImage <= sH)
        
        # Scale factor
        k = (oS[i+1] - oS[i]) / (sH-sL)

        oImage[idx] = k * (iImage[idx]-sL) + oS[i]

    return oImage

def gammaImage(iImage, iG):
    oImage = np.array(iImage, dtype=float)
    oImage = 255**(1-iG) * (iImage ** iG)
    return oImage


# VAJA 6

def getRadialValues(iXY, iCP):
    K = iCP.shape[0]

    # instanciranje izhodnih radialnih uteži
    oValue = np.zeros(K)

    x_i, y_i = iXY
    for k in range(K):
        x_k, y_k = iCP[k]

        # razdalja vhodne tocke do k-te kontrolne točke
        r = np.sqrt((x_i - x_k) ** 2 + (y_i -y_k) ** 2)

        # apliciranje radialne funkcije na r
        if r > 0:
            oValue[k] = -(r**2) * np.log(r)

    return oValue

def getParametrs(iType, scale = None, trans = None, rot = None, shear = None, orig_pts = None, mapped_pts = None):
    # default values
    oP = {}

    if iType == "affine":
        if scale is None:
            scale = [1,1]
        if trans is None:
            trans = [0,0]
        if rot is None:
            rot = 0
        if shear is None:
            shear = [0,0]

        Tk = np.array([
            [scale[0],0,0],
            [0,scale[1],0],
            [0,0,1]
        ])

        Tt = np.array([
            [1,0,trans[0]],
            [0,1,trans[1]],
            [0,0,1]
        ])

        phi = rot*np.pi / 180

        Tf = np.array([
            [np.cos(phi),-np.sin(phi),0],
            [np.sin(phi),np.cos(phi),0],
            [0,0,1]
        ])

        Tg = np.array([
            [1,shear[0],0],
            [shear[1],1,0],
            [0,0,1]
        ])        

        oP = Tg @ Tf @ Tt @ Tk

    elif iType == "radial":
        assert orig_pts is not None, "Manjkajo orig_pts"
        assert mapped_pts is not None, "Manjkajo mapped_pts"

        K = orig_pts.shape[0]

        UU = np.zeros((K,K), dtype=float)

        for i in range(K):
            UU[i,:] = getRadialValues(orig_pts[i,:], orig_pts)

        oP["alphas"] = np.linalg.solve(UU, mapped_pts[:,0])
        oP["betas"] = np.linalg.solve(UU, mapped_pts[:,1])
        oP["pts"] = orig_pts
        
    return oP


def transformImage(iType, iImage, iDim, iP, iBgr=0, iInterp=0):
    Y, X = iImage.shape
    dx, dy = iDim

    oImage = np.ones((Y,X))*iBgr

    for y in range(Y):
        for x in range(X):
            x_hat, y_hat = x*dx, y*dy
            
            if iType == "affine":
                x_hat, y_hat, _ = iP @ np.array([x_hat,y_hat,1])

            if iType == "radial":
                U = getRadialValues([x_hat,y_hat], iP["pts"])
                x_hat,y_hat = np.array([U @ iP["alphas"], U @ iP["betas"]])            

            x_hat, y_hat = x_hat/dx, y_hat/dy

            if iInterp == 0:
                x_hat, y_hat = round(x_hat), round(y_hat)
                if 0 <= x_hat < X and 0 <= y_hat < Y:
                    oImage[y, x] = iImage[y_hat,x_hat]    
    return oImage


