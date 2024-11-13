import numpy as np
import matplotlib.pyplot as plt

import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from LabVaje.OSV_lib import displayImage, getPlanerCrossSection, getPlanarProjection


def loadImage3D(iPath, iSize, iType):
    fid = open(iPath, "rb")
    im_shape = (iSize[1], iSize[0], iSize[2])
    oImage = np.ndarray(shape=im_shape, dtype=iType, buffer=fid.read(), order="F")
    fid.close()

    return oImage




if __name__=="__main__":
    imSize = [512, 58, 907]
    pxDim = [0.597656, 3, 0.597656]
    I = loadImage3D(r"LabVaje/Vaja4/data/spine-512x058x907-08bit.raw",
                    imSize,
                    np.uint8)
    print(I.shape)
    displayImage(I[:, 250, :], "Prerez")

    xc = 256
    sagCS, sagH, sagV = getPlanerCrossSection(I, pxDim, [1,0,0], xc)
    displayImage(sagCS, "Sagital crossection", sagH, sagV)

    xc = 35
    sagCS, sagH, sagV = getPlanerCrossSection(I, pxDim, [0,1,0], xc)
    displayImage(sagCS, "Coronal crossection", sagH, sagV)

    xc = 467
    sagCS, sagH, sagV = getPlanerCrossSection(I, pxDim, [0,0,1], xc)
    displayImage(sagCS, "Axial crossection", sagH, sagV)


    func = np.max
    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [1,0,0], func)
    displayImage(sagP, "Sagital projection (Function = MAX)", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0,1,0], func)
    displayImage(sagP, "Coronal projection (Function = MAX)", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0,0,1], func)
    displayImage(sagP, "Axial projection (Function = MAX)", sagH, sagV)

    func = np.average
    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [1,0,0], func)
    displayImage(sagP, "Sagital projection (Function = Average)", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0,1,0], func)
    displayImage(sagP, "Coronal projection (Function = Average)", sagH, sagV)

    [sagP, sagH, sagV] = getPlanarProjection(I, pxDim, [0,0,1], func)
    displayImage(sagP, "Axial projection (Function = Average)", sagH, sagV)