import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Naloga 1
    im = plt.imread('C:/Projects/OSV/LabVaje/Vaja1/data/lena-color.png')

    plt.figure()
    plt.imshow(im)
    #plt.show()
    plt.imsave('C:/Projects/OSV/LabVaje/Vaja1/data/lena-color_new.jpg', im)

# Naloga 2
def loadImage(iPath, iSize, iType):
    fid = open(iPath, 'rb')
    buffer = fid.read()
    buffer_len = len(np.frombuffer(buffer=buffer, dtype=iType))
    if buffer_len != np.prod(iSize):
        raise ValueError('Size of the input data does not match the specified size')
    else:
        oImage_Shape = (iSize[1],iSize[0])

    oImage = np.ndarray(oImage_Shape, dtype = iType, buffer = buffer, order='F')
    return oImage

# Naloga 3
def displayImage(iImage, iTitle = ''):
    fig = plt.figure()
    plt.title(iTitle)
    plt.imshow(iImage, 
               cmap='gray',
               vmin=0,
               vmax=255,
               aspect='equal')
    return fig

if __name__ == '__main__':
    image_2_gray = loadImage('C:/Projects/OSV/LabVaje/Vaja1/data/lena-gray-410x512-08bit.raw', (410,512), np.uint8)
    displayImage(image_2_gray, 'Lena')
    plt.show()