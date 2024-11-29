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
