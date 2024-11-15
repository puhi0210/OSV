import numpy as np
import matplotlib.pyplot as plt

import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from LabVaje.OSV_lib import loadImage, displayImage, scaleImage, windowImage, sectionalScaleImage, gammaImage

if __name__ == "__main__":
    I =loadImage("LabVaje/Vaja5/data/image-512x512-16bit.raw", [512,512], np.int16)
    displayImage(I, "Originalna slika")
    print (f"Input image:\tmin = {I.min()}, max = {I.max()}")
    
    sImage = scaleImage(I, -0.125, 256)
    displayImage(sImage, "Skalirana slika")
    print (f"Scaled image:\tmin = {sImage.min()}, max = {sImage.max()}")

    wImage = windowImage(sImage, 1000, 500)
    displayImage(wImage, "Oknjena skalirana slika")
    print (f"Windowed image:\tmin = {wImage.min()}, max = {wImage.max()}")

    sCP = np.array([[0,85],[85,0],[170,255],[255,170]])
    ssImage = sectionalScaleImage(wImage, sCP[:,0], sCP[:,1])
    displayImage(ssImage, "Odsekoma skalirana slika")
    print (f"Sectionali scaled image:\tmin = {ssImage.min()}, max = {ssImage.max()}")

    gImage = gammaImage(wImage, 5)
    displayImage(gImage, "Gamma preslikana slika")
    print (f"Gamma image:\tmin = {gImage.min()}, max = {gImage.max()}")
