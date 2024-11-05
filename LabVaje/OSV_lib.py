
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
def displayImage(iImage, iTitle = ''):
    fig = plt.figure()  # Ustvarimo figuro za prikaz slike
    plt.title(iTitle)   # Ustvarimo figuro za prikaz slike
    # Prikaz slike s sivinsko barvno karto in določenimi mejnimi vrednostmi
    plt.imshow(iImage, 
               cmap='gray',
               vmin=0,
               vmax=255,
               aspect='equal')
    return fig  # Vrne figuro


