"""Zagovor 2021-3, resitve"""
import numpy as np
import matplotlib.pyplot as plt
from vaja03.code.python.script import displayImage
from vaja05.code.python.script import thresholdImage
from vaja06.code.python.script import getParameters, transformImage
from vaja07.code.python.script import spatialFiltering


# 1. naloga
def color2grayscale(iImage):
    return iImage.mean(axis=2).astype(int)


if __name__ == "__main__":
    I = plt.imread("zagovor_2021_3/code/data/paris_map-807-421.png")
    displayImage(I, "Originalna slika")

    Igc = color2grayscale(I * 255)
    displayImage(Igc, "Sivinska slika")


def displayPoints(iXY, iMarker):
    plt.plot(iXY[:, 0], iXY[:, 1], iMarker, ms=5, lw=2)


def scale2range(img_np):
    im = np.asarray(img_np, dtype=float)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im


# 2. naloga
if __name__ == "__main__":
    # rocno izloceni tocki
    A = np.array([354, 156])
    B = np.array([400, 172])

    displayImage(Igc, "Sivinska slika")
    displayPoints(A.reshape(1, -1), "ro")
    displayPoints(B.reshape(1, -1), "bo")

    # izracun kota med vektorjema
    vec = A - B
    unit_vec = np.array([1, 0])
    phi = 180 - np.arccos((unit_vec @ vec) / np.linalg.norm(vec)) * 180 / np.pi

    # rotacija slike okoli tocke A
    T_center = getParameters("affine", trans=-A)
    T = getParameters("affine", rot=-phi)
    T_center_inv = getParameters("affine", trans=A)
    A = T_center_inv @ T @ T_center
    trI = transformImage("affine", Igc, [1, 1], np.linalg.inv(A), 0, 1)
    displayImage(scale2range(trI), "Poravnava slika")

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobelov x operator
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobelov y operator
    Sx = spatialFiltering("kernel", trI, Kx)
    Sy = spatialFiltering("kernel", trI, Ky)
    Sa = np.sqrt(Sx**2 + Sy**2)
    displayImage(scale2range(Sa), "Amplitudna slika gradienta")

    tI = thresholdImage(scale2range(Sa), 50)
    displayImage(tI, "Upragovljena slika")


# 4. naloga
def getCenterPointSquare(iImage, iLength):
    # velikost slike in inicializacija akumulatorja
    Y, X = iImage.shape
    oAcc = np.zeros((Y, X), dtype=int)

    # zanka cez vse slikovne elemente
    for y in range(Y):
        for x in range(X):
            # robna tocka
            if iImage[y, x]:
                for i_x in np.arange(-iLength / 2, iLength / 2 + 1):
                    for i_y in np.arange(-iLength / 2, iLength / 2 + 1):
                        # izracunaj koordinate na robovih kvadrata
                        xR = round(x + i_x)
                        yR = round(y + i_y)
                        # dodaj kvadrat v akumulator
                        if 0 < xR < X and 0 < yR < Y:
                            oAcc[yR, xR] += 1

    # koordinate sredisca kvadrata
    y0, x0 = np.unravel_index(np.argmax(oAcc, axis=None), oAcc.shape)
    oCenter = np.array([x0, y0])
    return oCenter, oAcc


if __name__ == "__main__":
    square_edge = 50
    oCenter, oAcc = getCenterPointSquare(tI, square_edge)
    displayImage(scale2range(oAcc), "Slika akumulatorja")

    # doloci vsa stiri oglisca kvadrata
    x_c, y_c = oCenter
    x1, y1 = x_c - square_edge / 2, y_c - square_edge / 2
    x2, y2 = x1 + square_edge, y1
    x3, y3 = x1, y1 + square_edge
    x4, y4 = x1 + square_edge, y1 + square_edge

    # prikazi poravnano sliko in kvadrat ter sredisce
    displayImage(scale2range(trI), "Koncni rezultat")
    displayPoints(oCenter.reshape(1, -1), "ro")
    plt.plot([x1, x2], [y1, y2], "r", linewidth=2)
    plt.plot([x1, x3], [y1, y3], "r", linewidth=2)
    plt.plot([x2, x4], [y2, y4], "r", linewidth=2)
    plt.plot([x3, x4], [y3, y4], "r", linewidth=2)
