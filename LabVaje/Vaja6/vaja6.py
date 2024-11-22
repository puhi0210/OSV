import numpy as np
import matplotlib.pyplot as plt

import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from LabVaje.OSV_lib import loadImage, displayImage, getParametrs, transformImage

if __name__ == "__main__":
    imSize = [256,512]
    pxDim = [2,1]

    gX = np.arange(imSize[0])*pxDim[0]
    gY = np.arange(imSize[1])*pxDim[1]
    I = loadImage("LabVaje/Vaja6/data/lena-256x512-08bit.raw", imSize, np.uint8)
    displayImage(I, "Originalna slika", gX,gY)

    T = getParametrs("affine", rot=30)
    print(T)

    brg = 63

    tImage = transformImage("affine", I, pxDim, np.linalg.inv(T), iBgr=brg)
    displayImage(tImage, "Affina preslikava", gX,gY)

    xy = np.array([[0,0],[511,0],[0,511],[511,511]])
    uv = np.array([[0,0],[511,0],[0,511],[255,255]])
    P = getParametrs("radial", orig_pts=xy, mapped_pts=uv)
    print(P)
    rImage = transformImage("radial", I, pxDim, P, iBgr=brg)
    displayImage(rImage, "Radialna preslikava", gX,gY)
