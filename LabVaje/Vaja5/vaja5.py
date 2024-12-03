import numpy as np
import matplotlib.pyplot as plt

import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from LabVaje.OSV_lib import loadImage, displayImage, scaleImage, windowImage, sectionalScaleImage, gammaImage, thresholdImage

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

    # GRADIVO
    # Naloga 2
    tImage = thresholdImage(wImage, 127)
    displayImage(tImage, "Slika po upragovanju")
    print (f"Gamma image:\tmin = {tImage.min()}, max = {tImage.max()}")

    # Naloga 3
    t_values = np.arange(wImage.min(), wImage.max() + 1, 1)
    sg_values = []

    for t in t_values:
        tImage = thresholdImage(wImage, t)
        sg_values.append(np.sum(tImage == 0))

    plt.plot(t_values, sg_values)
    plt.xlabel('Pragovna vrednost (t)')
    plt.ylabel('Število slikovnih elementov s sivinsko vrednostjo 0')
    plt.title('Pragovna funkcija')
    plt.grid(True)
    plt.show()
