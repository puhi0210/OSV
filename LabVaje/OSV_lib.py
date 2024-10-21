
#   Knjižnica funkcij za laboratorijske vaje pri predmetu OSV


import matplotlib.pyplot as plt
import numpy as np


def loadImage(iPath, iSize, iType):
    fid = open(iPath, 'rb')
    buffer = fid.read()
    buffer_len = len(np.frombuffer(buffer=buffer, dtype=iType))
    if buffer_len != np.prod(iSize):
        raise ValueError('Size of the input data does not match the specified size')
    else:
        oImage_Shape = (iSize[1],iSize[0])

    oImage = np.ndarray(oImage_Shape, dtype = iType, buffer = buffer, order='F')
    return oImage

def displayImage(iImage, iTitle = ''):
    fig = plt.figure()
    plt.title(iTitle)
    plt.imshow(iImage, 
               cmap='gray',
               vmin=0,
               vmax=255,
               aspect='equal')
    return fig
