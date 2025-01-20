from pickletools import uint8
from matplotlib import pyplot as plt
import numpy as np
from Vaja_2.skripta import loadImage
from Vaja_4.skripta import displayImage3D as displayImage


I = plt.imread('Zagovori/Primer_zagovora3/planina-uint8.jpeg')
#I_sm = I*255
#I_255 = np.dot(I, 255)
print(I.shape, I.dtype)


# 1. Naloga
def standardizeImage(iImage):
    oImage = np.zeros(iImage.shape).astype(float)
    R = iImage[:,:,0]
    G = iImage[:,:,1]
    B = iImage[:,:,2]

    r_pov = np.mean(R)
    r_std = np.std(R)
    red = (R - r_pov)/r_std

    g_pov = np.mean(G)
    g_std = np.std(G)
    green = (G - g_pov)/g_std

    b_pov = np.mean(B)
    b_std = np.mean(B)
    blue = (B - b_pov)/b_std

    oImage[:,:,0] = red
    oImage[:,:,1] = green
    oImage[:,:,2] = blue

    return oImage

# 2. Naloga
def rgb2hsl(iRGB):
    Y,X,C = iRGB.shape
    oHSL = np.zeros(iRGB.shape).astype(float)

    for i in range(Y):
        for j in range(X):
            kanal = iRGB[i][j]
            v = np.max((kanal[0], kanal[1], kanal[2]))
            v_min = np.min((kanal[0], kanal[1], kanal[2]))
            ci = v - v_min
            li = v - (ci/2)

            H = 0
            if v == kanal[0]: #red
                H = (60 * (((kanal[1]-kanal[2])/ci)))
                #print("H je: ", H)
            elif v == kanal[1]: #green
                H = (60 * ((2+(kanal[2]-kanal[0])/ci)))
                #print("H je: ", H)
            elif v == kanal[2]: #blue
                H = (60 * ((4+(kanal[0]-kanal[1])/ci)))
                #print("H je: ", H)
            elif ci == 0:
                H = 0
                #print("H je: ", H)

            #oHSL[i,j, 0] = H

            if (li == 0) or (li == 1):
                s = 0
            else:
                s = (ci/(1-(np.abs(2*v-ci-1))))

            oHSL[i,j, 1] = s
            oHSL[i,j, 2] = v
            oHSL[i,j, 0] = H
            
    return oHSL

# 4. Naloga
def hsl2rgb(iHSL):
    Y,X,C = iHSL.shape
    oRGB = np.copy(iHSL)
    tempRGB = np.zeros(iHSL.shape).astype(float)

    for i in range(Y):
        for j in range(X):
            # H, S, Li
            kanal = iHSL[i][j]
            ci = (1-(np.abs(2*kanal[2]-1))*kanal[1])
            H = kanal[0]/60

            xi = ci * (1-(np.abs(H % 2)-1))
            
            if 0 <= H < 1:
                tempRGB[i, j, 0] = ci
                tempRGB[i, j, 1] = xi
                tempRGB[i, j, 2] = 0.0
            elif 1 <= H < 2: 
                tempRGB[i, j, 0] = xi
                tempRGB[i, j, 1] = ci
                tempRGB[i, j, 2] = 0.0
            elif 2 <= H < 3:  
                tempRGB[i, j, 0] = 0.0
                tempRGB[i, j, 1] = ci
                tempRGB[i, j, 2] = xi
            elif 3 <= H < 4:
                tempRGB[i, j, 0] = 0.0
                tempRGB[i, j, 1] = xi
                tempRGB[i, j, 2] = ci
            elif 4 <= H < 5:
                tempRGB[i, j, 0] = xi
                tempRGB[i, j, 1] = 0.0
                tempRGB[i, j, 2] = ci
            elif 5 <= H < 6:
                tempRGB[i, j, 0] = ci
                tempRGB[i, j, 1] = 0.0
                tempRGB[i, j, 2] = xi

            mi = kanal[2] - ci / 2

            oRGB[i, j, 0] = (tempRGB[i, j, 0] + mi)
            oRGB[i, j, 1] = (tempRGB[i, j, 1] + mi)
            oRGB[i, j, 2] = (tempRGB[i, j, 2] + mi)

    return oRGB



if __name__ == "__main__":
    plt.figure() #sliko damo v figure, podobno kot Matlab
    plt.imshow(I) #sliko prikažemo


    plt.figure
    plt.imshow(standardizeImage(I))

    I_std = standardizeImage(I)

    # 2. Naloga
    # 
    plt.figure
    I_hsl = rgb2hsl(I_std)
    displayImage(I_hsl)

    # 3. Naloga
    # I_hsl
    h_slice = I_hsl[:,:,0]

    Y,X = h_slice.shape
    new_slice = np.zeros((Y,X)).astype(float)

    for y in range(Y):
        for x in range(X):
            if h_slice[y][x] >= 100 and h_slice[y][x] < 250:
                new_slice[y][x] = h_slice[y][x]/5
    #print(new_slice)
                
    I_hsl_ref = np.copy(I_hsl)
                
    I_hsl_ref[:,:,0] = new_slice
    displayImage(I_hsl_ref)

    I_hsl2rgb = hsl2rgb(I_hsl)
    displayImage(I_hsl2rgb)

    