import numpy as np
import matplotlib.pyplot as plt

import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from LabVaje.OSV_lib import loadImage, displayImage, spatialFiltering, changeSpatialDomain

if __name__ == "__main__":
    I = loadImage("LabVaje/Vaja7/data/cameraman-256x256-08bit.raw", [256,256], np.uint8)
    displayImage(I,"Originalna slika")


    Kernel = np.array([
        [1,1,1],
        [1,-8,1],
        [1,1,1]
    ])
    KImage = spatialFiltering("kernel", I, iFilter=Kernel)
    displayImage(KImage,"Filtrirana slika z laplacovim filtrom")
    
    SImage = spatialFiltering("statistical", I, iFilter=np.zeros((30,30)), iStatFunc=np.median)
    displayImage(SImage,"Statisticno filtrirana slika: mediana")

    MKernel = np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0]
    ])
    MImage = spatialFiltering("morphological", I, iFilter=MKernel, iMorphOp="erosion")
    displayImage(MImage,"Morfološko filtriranje: erozija")

    PaddedImage = changeSpatialDomain("enlarge", I, 30, 30)
    displayImage(PaddedImage,"Razširjena sliak z vrednostjo 0")

    # Sobel operator
    sobelX = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])

    sobelY = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])

    sxImage = spatialFiltering("kernel", I, iFilter=sobelX)
    displayImage(sxImage,"Filtrirana slika s Soblovim operatorjem X")

    syImage = spatialFiltering("kernel", I, iFilter=sobelY)
    displayImage(syImage,"Filtrirana slika s Soblovim operatorjem Y")

    saImage = np.zeros(sxImage.shape)

    for i in range(sxImage.shape[0]):
        for j in range(sxImage.shape[1]):
            saImage[j,i] = np.sqrt(sxImage[j,i]**2 + syImage[j,i]**2)

            saImage[j,i] = (saImage[j,i] * 255)/np.sqrt(255**2 + 255**2)

    displayImage(saImage,"Amplitudna filtrirana slika s Soblovim operatorjem")

    sphiImage = np.zeros(sxImage.shape)
    for i in range(sxImage.shape[0]):
        for j in range(sxImage.shape[1]):
            sphiImage[j,i] = np.arctan2(syImage[j,i], sxImage[j,i])
            sphiImage[j,i] = (sphiImage[j,i] * 255)/(2*np.pi)

    displayImage(sphiImage,"Filtrirana slika s Soblovim operatorjem: kotna slika")

    # Gaussov filter
    c = 2
    gauss = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            gauss[i,j] = np.exp(-((i-1)**2 + (j-1)**2)/(2*c**2))/(2*np.pi*c**2)
    
    gauss = gauss/np.sum(gauss)

    gImage = spatialFiltering("kernel", I, iFilter=gauss)
    displayImage(gImage,"Filtrirana slika z Gaussovim filtrom")

    mask = np.zeros(I.shape)

    mask = I - gImage

    mask = mask * 255/np.max(mask)
    displayImage(mask,"Razlika med originalno in filtrirano sliko")


    PaddedImage = changeSpatialDomain("enlarge", I, 128, 384, iMode="extrapolation")
    displayImage(PaddedImage,"Ekstrapolirana slika")
    
    PaddedImage = changeSpatialDomain("enlarge", I, 128, 384, iMode="reflection")
    displayImage(PaddedImage,"Zrcaljena slika")

    PaddedImage = changeSpatialDomain("enlarge", I, 128, 384, iMode="period")
    displayImage(PaddedImage,"Periodicna slika")