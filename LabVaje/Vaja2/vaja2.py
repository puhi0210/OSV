import numpy as np
import matplotlib.pyplot as plt

from OSV_lib import displayImage, loadImage

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

if __name__ == "__main__":
    image_equalized = equalizeHistogram(image)
    displayImage(image_equalized, "slika z izravnanim histogramom")
    hist, prob, CDF, levels = computeHistogram(image_equalized)
    displayHistogram(hist, levels, "histogram")    
    displayHistogram(prob, levels, "normaliziran histogram")
    displayHistogram(CDF, levels, "CDF histogram")