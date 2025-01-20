"""Zagovor 2021-1, resitve"""
import numpy as np
from vaja01.code.python.script import loadImage
from vaja03.code.python.script import displayImage
from vaja06.code.python.script import getParameters, transformImage


## 1. NALOGA
def getBoundaryIndices(iImage, iAxis):
    proj = np.sum(iImage, axis=iAxis).astype(bool)
    i_start = np.argmax(proj)
    i_end = np.argmax(proj[::-1])
    i_end = len(proj) - i_end - 1
    return i_start, i_end


if __name__ == "__main__":
    I = loadImage(
        "zagovor_2021_1/code/data/rose-366-366-08bit.raw", [366, 366], np.uint8
    )
    displayImage(I, "Originalna slika")
    # indeksi stolpcev
    xStart, xEnd = getBoundaryIndices(255 - I, 0)
    # indeksi vrstic
    yStart, yEnd = getBoundaryIndices(255 - I, 1)

    # izrez slike
    Icropped = I[yStart:yEnd, xStart:xEnd]
    displayImage(Icropped, "Cropped image")

    # velikost izrezane slike
    Ycropped, Xcropped = Icropped.shape

## 2. NALOGA
if __name__ == "__main__":
    # izloci zadnjo vrstico izrezane slike in vektor pretvori v matriko velikosti 1xN
    last_row_2d = Icropped[-1, :].reshape(1, -1)
    # izracunaj indekse stolpcev, ki omejujejo stebla, na invertirani sliki
    xx_1_lower, xx_2_lower = getBoundaryIndices(255 - last_row_2d, 0)
    # izracunaj sredino stebla
    xx_middle_lower = np.mean([xx_1_lower, xx_2_lower]).astype(int)

    # doloci vrstico, ki ustreza cetrtini slike v y osi
    yy_upper = int(Ycropped / 4)
    # izloci vektor intenzitet na tej vrstici in ga pretvori v matriko velikosti 1xN
    quater = Icropped[int(Ycropped / 4), :].reshape(1, -1)
    # doloci indekse stolpcev, ki omejujejo cvet, na invertirani sliki
    xx_1_upper, xx_2_upper = getBoundaryIndices(255 - quater, 0)

    centerStebla = np.array([xx_middle_lower, Ycropped])
    centerSlike = (np.array([Xcropped, Ycropped]) - 1) / 2
    leftCorner = np.array([xx_1_upper, yy_upper])
    rightCorner = np.array([xx_2_upper, yy_upper])

    # izracun kota fi
    a = leftCorner - centerStebla
    b = rightCorner - centerStebla
    fi_rad = np.arccos(
        (a * b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))
    )
    fi_deg = fi_rad * 180 / np.pi


def expandImage(iImage):
    Y1, X1 = iImage.shape
    oImage = 255 * np.ones((2 * Y1, 2 * Y1))
    Y2, X2 = oImage.shape
    center2 = (np.array([X2, Y2]) - 1) / 2
    x_tmp = int(center2[1] - X1 / 2)
    oImage[0:Y1, x_tmp : x_tmp + X1] = iImage
    return oImage


# naloga 3: klic funkcije za povecanje slike
if __name__ == "__main__":
    Ienlarged = expandImage(Icropped)
    displayImage(Ienlarged, "Povecana slika")


def createRotatedPattern(iImage, iAngle):
    Y, X = iImage.shape

    # izracun centra slike
    imgCenter = (np.array([X, Y]) - 1) / 2

    # inicializacija izhodne slike
    oImage = np.zeros_like(iImage, dtype=float)
    # matrika za translacijo v izhodisce
    Ttrans = getParameters(
        "affine", scale=[1, 1], trans=-imgCenter, rot=0, shear=[0, 0]
    )
    # matrika za translacijo nazaj
    Ttrans_inv = np.linalg.inv(Ttrans)

    # priprava vektorja kotov, zacni pri 0 in koncaj pri 360 - iAngle in razdeli na int(360/iAngle) delov
    angles = np.linspace(0, 360 - iAngle, int(360 / iAngle))
    for angle in angles:
        # trenutna matrika rotacije
        Trot = getParameters(
            "affine", scale=[1, 1], trans=[0, 0], rot=angle, shear=[0, 0]
        )
        # upostevaj naj bo rotacija okrog centra slike
        T = Ttrans_inv @ Trot @ Ttrans
        tmp = transformImage(
            "affine", iImage, iDim=[1, 1], iP=np.linalg.inv(T), iBgr=255, iInterp=0
        )
        # pristej k izhodni sliki
        oImage += tmp

    # normalizacija na interval [0, 255]
    oImage -= oImage.min()
    oImage /= oImage.max()
    oImage *= 255
    return oImage


# naloga 4: klic funkcije za izracun koncnega rotiranega vzorca
if __name__ == "__main__":
    oImage = createRotatedPattern(Ienlarged, fi_deg)
    displayImage(oImage, "Final image")
