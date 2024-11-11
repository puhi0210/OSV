import numpy as np
import matplotlib.pyplot as plt
import math
import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from LabVaje.OSV_lib import displayImage, loadImage

if __name__ == "__main__":
    image = loadImage("/home/miha/Projects/OSV/LabVaje/Vaja2/valley-1024x683-08bit.raw",
                      (1024, 683), np.uint8)
    displayImage(image, "originalna slika")


def computeHistogram(iImage):
    nBits = int(np.log2(iImage.max()))+1

    oLevels = np.arange(0, 2 ** nBits, 1)
    
    iImage = iImage.astype(np.uint8)

    oHist = np.zeros(len(oLevels))

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            oHist[iImage[y,x]] = oHist[iImage[y,x]] + 1

    oProb = oHist / iImage.size

    oCDF = np.zeros_like(oHist)

    for i in range(len(oProb)):
        oCDF[i] = oProb[:i+1].sum()

    return oHist, oProb, oCDF, oLevels


def displayHistogram(iHist, iLevels, iTitle):
    plt.figure()
    plt.title(iTitle)
    plt.bar(iLevels, iHist, width=1, edgecolor="darkred", color="red")
    plt.xlim((iLevels.min(), iLevels.max()))
    plt.ylim((0, 1.05 * iHist.max()))
    plt.show()


    
if __name__ == "__main__":
    hist, prob, CDF, levels = computeHistogram(image)
    displayHistogram(hist, levels, "histogram")    
    displayHistogram(prob, levels, "normaliziran histogram")
    displayHistogram(CDF, levels, "CDF histogram")



def equalizeHistogram(iImage):
    _, _, CDF, _ = computeHistogram(iImage)

    nBits = int(np.log2(iImage.max()))+1

    max_intensity = 2 ** nBits + 1

    oImage = np.zeros_like(iImage)

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            old_intensity = iImage[y,x]
            new_intensity = np.floor(CDF[old_intensity] * max_intensity)
            oImage[y,x] = new_intensity

    return oImage

if __name__ == "__main__" :
    image_equalized = equalizeHistogram(image)
    displayImage(image_equalized, "slika z izravnanim histogramom")
    hist, prob, CDF, levels = computeHistogram(image_equalized)
    displayHistogram(hist, levels, "histogram")    
    displayHistogram(prob, levels, "normaliziran histogram")
    displayHistogram(CDF, levels, "CDF histogram")


def computeEntropy(iImage):
    _, imgProb, _, _ = computeHistogram(iImage)

    oEntropy = 0

    for i in range(len(imgProb)):
        if imgProb[i] != 0:
            oEntropy += imgProb[i]* np.log2(imgProb[i]) *(-1)

    return oEntropy


if __name__ == "__main__":
    print('Entrophy of an image is ', computeEntropy(image))



def addNoise(iImage, iStd):
    oNoise = np.random.randn(iImage.shape[0],iImage.shape[1])* iStd + iStd 
    

    oImage = iImage + (oNoise - iStd)
    
    # Poskrbi, da vrednosti ostanejo v intervalih 0-255 
    oImage = np.clip(oImage, 0, 255)

    return oImage, oNoise

if __name__ == "__main__":
    std = [2, 5, 10, 25]
    for i in range(len(std)):
        noisyImage, noise = addNoise(image,std[i])
        displayImage(noisyImage, ("Slika z dodanim šumom(standardni odklon je  "+str(std[i])+")"))
        displayImage(noise, ("Slika Šuma (standardni odklon je "+str(std[i])+")"))
        