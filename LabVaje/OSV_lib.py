
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


