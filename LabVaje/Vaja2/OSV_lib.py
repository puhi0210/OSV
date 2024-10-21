import matplotlib.pyplot as plt
import numpy as np

# VAJA 1

def loadImage(iPath, iSize, iType):
    fid = open(iPath, 'rb')   # odpremo tunel in definiramo path do slike . . . 'rb' pomeni read in binary mode (odpre file na iPath za branje v binary modu)
    buffer = fid.read()  # v buffer shranimo spremenljivko, zato da jo bomo lahko prebrali v nadaljevanju
    buffer_len = len(np.frombuffer(buffer = buffer, dtype = iType))

    if buffer_len != np.prod(iSize):
        raise ValueError('Size of the input data does not match the specified size.')
    else:
        oImage_shape = (iSize[1], iSize[0])
    
    oImage = np.ndarray(oImage_shape, dtype = iType, buffer = buffer, order = 'F')  # order F pomeni v kakšnem stilu je zapisan v bufferju podatek
    return oImage

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# NALOGA 3

def displayImage(iImage, iTitle = ''):
    fig = plt.figure()
    plt.title(iTitle)
    
    
    ex = 40             # extend...tega ubistvu ne rabimo in je odveč
    plt.imshow(iImage, 
               cmap = 'gray', 
               vmin = 0, # vmin in vmax je oknenje (če so večje bo ostalo 255, če bo manjše bo ostalo 0)
               vmax = 255,
               aspect = 'equal', 
               # extent = (0-ex,
               #          iImage.shape[1]-ex,
               #          iImage.shape[0]-ex,
               #          0-ex)
               )
    
    plt.show()
    return fig

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# DODATNI DEL

def saveImage(iImage, iPath, iType):
    ImageFormat = iImage.astype(iType)   # convertanje pixlov slike v ustrezen data type (v našem primeru uint8)
    newImage = open(iPath, 'wb')         # 'wb' to pomeni write binary (open a file for writing in binary)
    newImage.write(ImageFormat.tobytes(order = 'F'))   # order = F ... fortran mode zapis; da bojo shranjen slike kompatibilne s prejšnjimi funkcijami
                                                       # converta numpy array (v kateri je data od slike) v bajtni object (raw pixel data --> zaporedje bajtov)
    newImage.close()    # naj bi se file po writanju raw bajtov avtomatsko zaprl