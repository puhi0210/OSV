from pickletools import uint8
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from Vaja_2.skripta import loadImage
from Vaja_4.skripta import displayImage3D as displayImage

I = plt.imread('Zagovori/Primer_zagovora2/paris_map-807-421.png')
#I_sm = I*255
I_255 = np.dot(I, 255)
print(I_255.shape, I_255.dtype)



# 1. Naloga
def color2grayscale ( iImage ) :
    oImage = np.zeros((np.shape(iImage)[0], np.shape(iImage)[1]), dtype = np.float64) #ustvarimo blank uint8 matriko velikosti pngja
    #R, G, B = ((iImage[0].astype(float)), (iImage[1].astype(float)), (iImage[2]).astype(float)) #ven dobimo kanale
    
    #Gr = (R + G + B)/3

    oImage = np.sum(I_255, axis=-1)/3
    oImage = oImage.astype(int)

    return oImage

# 2. Naloga, koda naloge za transformImage

def getRadialValue(iXY, pts):
    K = pts.shape[0]

    xi, yi = iXY
    oValue = np.zeros(K, dtype=float)

    for k in range(K): #sprehajamo se čez kontrolne točke, računamo razdalje od y do y1, y, y2,...
        xk, yk = pts[k, :]
        r = np.sqrt((xi - xk)**2 + (yi-yk)**2) #evklidska razdalja med y in yk

        if r > 0:
            oValue[k] = -r**2 * np.log(r)
    return oValue

def getCenter(iImage, px_dim):
    center = (iImage.shape[0]/2*px_dim[1], iImage.shape[1]  /2*px_dim[0])
    return center

def getParameters ( iType , scale = None , trans = None , rot = None ,
    shear = None , orig_pts = None , mapped_pts = None, center = None) :

    if iType == "affine": #afina preslikava
        if scale is None:
            scale = [1, 1]
        if trans is None:
            trans = [0, 0] #navodila, tx in ty sta 0
        if rot is None: 
            rot = 0 #podan v stopinjah
        if shear is None:
            shear = [0, 0]
        #np.eye() nam naredi matriko ki ima po diagonali enke (identiteta)
        Tk = np.eye(3)
        Tk[0, 0] = scale[0]
        Tk[1, 1] = scale[1] #glej matriko v navodilih!!!!!!!!!¨

        Tt = np.eye(3)
        Tt[:2, 2] = trans #:2 gre do druge vrstice

        phi = rot * np.pi/180 #kot v radiane ker uporabljamo numpy 
        Tr = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ])

        Tg = np.eye(3)
        Tg[0, 1] = shear[0]
        Tg[1, 0] = shear[1]
        
        #4. Naloga, preslikava okrog "centra"
        if center:
            Tc_pos = np.array([ #https://math.stackexchange.com/questions/2093314/rotation-matrix-of-rotation-around-a-point-other-than-the-origin
                [1, 0, -center[0]],
                [0, 1, -center[1]],
                [0, 0, 1]
            ])
            
            Tc_neg = np.array([
                [1, 0, center[0]],
                [0, 1, center[1]],
                [0, 0, 1]
            ])

            oP = Tc_neg @ Tg @ Tr @ Tt @ Tk @ Tc_pos #zmnožiš vse matrike
        else:
            oP = Tg @ Tr @ Tt @ Tk #matrično množenje

    elif iType == "radial": #radialna preslikava
        K = orig_pts.shape[0]
        UU = np.zeros((K,K), dtype = float)
        for k in range (K):
            UU[k, :] = getRadialValue(orig_pts[k, :], orig_pts) #orig_pts so kontrolne točke originalnega koordinatnega sistema

        oP = {}
        oP['alphas'] = np.linalg.solve(UU, mapped_pts[:,0]) #alfe dobimo ven iz enačbe z funkcijo linagl solve
        oP['betas'] = np.linalg.solve(UU, mapped_pts[:,1]) #bete dobimo isto
        oP['pts'] = orig_pts #dodamo matriko kontrolnih točk, da nam funkcija pripravi vse potrebno za izračun
    else:
        raise NotImplementedError('Neznam iType??')

    return oP

def transformImage(iType , iImage , iDim , iP , iInterp, iBgr = 0):
    Y, X = iImage.shape
    
    oImage = np.ones((Y, X)) * iBgr

    dx, dy = iDim

    for y in range(Y):
        for x in range(X):
            x_hat, y_hat = x*dx, y*dy
            if iType == "affine":
                x_hat, y_hat, _ = iP @ np.array([x_hat, y_hat, 1])
            elif iType == "radial":
                U = getRadialValue([x_hat, y_hat], iP['pts'])
                x_hat, y_hat = U @ iP['alphas'], U @ iP['betas']

            x_hat, y_hat = x_hat/dx, y_hat/dy

            if iInterp == 0:
                x_hat, y_hat = round(x_hat), round(y_hat)
                
                if 0 <= x_hat < X and 0 <= y_hat < Y:
                    oImage[y,x] = iImage[y_hat, x_hat]
            elif iInterp == 1: #interpolacija II. reda
                if 0 <= x_hat < X and 0 <= y_hat < Y: #obvezno isti if stavek, drugače gre izven matrike
                    #koda 3. vaje, test.py
                    x_a, y_a = int(x_hat), int(y_hat)
                    x_b, y_b = x_a + 1, y_a
                    x_c, y_c = x_a, y_a + 1
                    x_d, y_d = x_a + 1, y_a +1

                    #obvezno min(x_a, X-1) drugače gre matrika out of bounds!!!
                    sa = iImage[y_a, x_a]
                    sb = iImage[y_b, min(x_b, X - 1)]
                    sc = iImage[min(y_c, Y - 1), x_c]
                    sd = iImage[min(y_d, Y - 1), min(x_d, X - 1)] 

                    a = abs((x_d - x_hat) * (y_d - y_hat))
                    b = abs((x_hat - x_c) * (y_hat - y_c))
                    c = abs((x_b - x_hat) * (y_b - y_hat))
                    d = abs((x_hat - x_a) * (y_hat - y_a))

                    s = sa * a + sb * b + sc * c + sd * d   
                    oImage[y,x] = s
    return oImage

# 3. naloga: Sobel X in Y 

def changeSpatialDomain ( iType , iImage , iX , iY , iMode = None , iBgr =0) :
    Y,X = iImage.shape
    #iX število stoplcev ki jih dodamo, iY število vrstic
    if iType == 'enlarge':
        if iMode is None:
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage #glej sliko
        #elif doma pomoč z np.vstack in np.hstack
        elif iMode == "constant":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) + iBgr
            oImage[iY:Y + iY, iX:X + iX] = iImage
        elif iMode == "extrapolation":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage  #input slika gre na sredino nove slike

            oImage[:iY, iX:iX + X] = iImage[0, :] #SEVER
            oImage[iY + Y:, iX:iX + X] = iImage[-1, :] #JUG

            oImage[:, :iX] = np.tile(oImage[:, iX:iX + 1], (1, iX))  #VZHOD
            oImage[:, iX + X:] = np.tile(oImage[:, iX + X - 1:iX + X], (1, iX))#ZAHOD
        elif iMode == "reflection":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage 

            # Za zgornji rob
            for i in range(iY):
                if not i // Y % 2:
                    oImage[iY - i, :] = oImage[iY + i % Y, :]
                else:
                    oImage[iY - i, :] = oImage[iY + Y - i % Y, :]

            # Za levi rob
            for i in range(iX):
                if not i // X % 2:
                    oImage[:, iX - i] = oImage[:, iX + i % X]
                else:
                    oImage[:, iX - i] = oImage[:, iX + X - i % X]

            # Za desni rob
            for i in range(iX):
                oImage[:, iX + X + i] = oImage[:, iX + X -1 -i]
            
            # Za spodnji rob
            for i in range(iY):
                oImage[iY + Y + i, :] = oImage[iY + Y -1 -i, :]
        elif iMode == "period":
            oImage = np.zeros((Y + 2*iY, X + 2*iX)) #na vsako stran prištejemo iX
            oImage[iY:Y + iY, iX:X + iX] = iImage 

            # Za zgornji rob
            for i in range(iY):
                oImage[iY - i, :] = oImage[iY + Y - i % Y, :]

            # Za spodnji rob
            for i in range(iY + Y, 2 * iY + Y):
                oImage[i, :] = oImage[i - Y, :]

            # Za spodnji rob
            for i in range(iX):
                oImage[:, iX - i] = oImage[:, iX + X - i % X]
            
            # Za desni rob
            for i in range(iX + X, 2 * iX + X):
                oImage[:, i] = oImage[:, i - X]
        else:
            raise ValueError("napačen iMode")
    elif iType == 'reduce':
        oImage = iImage[iY:Y - iY, iX:X - iX] #lih obratno
    else:
        raise ValueError("napačen iType")
        
    return oImage

def spatialFiltering ( iType , iImage , iFilter , iStatFunc = None, iMorphOp = None ) :
    M, N = iFilter.shape #dimneziji filtra
    m = int((M-1)/2) #koliko praznih vrstic/stolpcev imamo
    n = int((N-1)/2)

    iImage = changeSpatialDomain('enlarge', iImage, iX = n, iY = m)

    Y, X = iImage.shape #dobimo dimnezijo slike
    oImage = np.zeros((Y,X), dtype = float) #inicializiramo oImage, velikosti slike, same ničle

    for y in range(n, Y-n): #do Y-n, ker gre do konca slike minus polovica filtra
        for x in range(m, X-m):
            patch = iImage[y-n:y+n+1, x-m:x+m+1] #tole je filter v sliki, kje se nahaja. +1 je notri ker python z : gleda do (ne vključno)

            if iType == 'kernel':
                oImage[y, x] = np.sum(patch * iFilter) #filter na sliki množimo z parametri filtra, TRANSPONIRANJE ZA MNOŽENJE Z MATRIKAMI KI NISO NxN
            elif iType == 'kernel_mismatch':
                oImage[y, x] = np.sum(patch * iFilter.T)
            elif iType == 'statistical':
                oImage[y,x] = iStatFunc(patch) #iStatFunc gre notr np.mean ipd, poglej prejšne vaje
            elif iType == 'morphological':
                R = patch[iFilter != 0] #pogleda kje v matriki so vrednosti 1 in filtrira
                if iMorphOp == 'erosion':
                    oImage[y,x] = R.min() #np.min(R)
                elif iMorphOp == 'dilation':
                    oImage[y,x] = R.max()
                else:
                    raise NotImplementedError('Napačen iMorphOp')
            else:
                raise NotImplementedError('Napačen iType')

    oImage = changeSpatialDomain('reduce', oImage, iX = m, iY = n)
    return oImage

def tresholdImage(iImage, iT):
    oImage = iImage.astype(float).copy()
    oImage[iImage <= iT] = 0 #formula v navodilih
    oImage[iImage > iT] = 255

    return oImage

Sb_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

Sb_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])


# 4. naloga, Huhhgh

# velikost stranice kvadrata:
#pogledam vrstico slike 160 (npr) in v njej izpišem vrednosti, kjer je vrednost drugačna kot 0
# np.where(iImage[160,:] > 0)
# to mi da array (array([  0, 352, 353, 355, 356, 402, 404, 405, 806]),)
# tu lahko vidimo da se kvadrat začne v 352 in gre do 405, kar pomeni da je dolžina stranice
# a = 53



def getSquareCenterPoint(iImage, iLength):
    #oAcc = np.zeros((iImage.shape[0], iImage.shape[1])) #array.size v np poda število elementov v arrayu
    Y, X = iImage.shape
    oAcc = np.zeros((Y,X))

    for y in range(Y):
        for x in range(X):
            if iImage[y,x] > 0:
                for fi in range (360): #sprehodimo se po obodu krožnice, na vsaki točki (360) narišemo svojo krožnico in gledamo presečišča
                    fi_rad = fi * np.pi / 180
                    x_fi = int((y + (iLength/2 * np.sin(fi_rad))))
                    y_fi = int((x + (iLength/2 * np.cos(fi_rad))))
                               
                    if (0 <= x_fi < Y and 0 <= y_fi < X): #gledamo presečišča, aka enačimo z nič, <= ker drugače ignorira
                        oAcc[x_fi, y_fi] += 1
    oCenter = 0
    max_value = 0
    for i in range(len(oAcc)):
        for j in range(len(oAcc[i])):
            if oAcc[i][j] > max_value:
                max_value = oAcc[i][j]
                oCenter = (i, j)        
    print("Središče kvadrata se nahaja v: ", oCenter)
    print("Maksimalna vrednost akumulatorja je: ", np.max(oAcc))
    displayImage(oAcc)
    return oAcc, oCenter

if __name__ == "__main__":
    # 1. Naloga
    plt.figure() #sliko damo v figure, podobno kot Matlab
    plt.imshow(I) #sliko prikažemo


    print(I.shape, I.dtype) #3 tu pomeni barvno globino, slika je float32

    I_grey = color2grayscale(I_255)
    
    print(I_grey.shape, I_grey.dtype)
    plt.figure()
    plt.imshow(I_grey, cmap="gray")

    #ročno določeni točki
    A = np.array([354, 156])
    B = np.array([400, 172])

    # Vektorji
    AB = A-B
    B = np.array([1,0])
    abs_AB = np.sqrt(np.dot(AB,AB))
    abs_B = np.sqrt(np.dot(B, B))
    #kot = np.degrees(np.arccos((np.dot(AB,B)/abs_AB*abs_B)))
    kot = np.degrees(np.arccos((np.dot(AB,B)/(abs_AB*abs_B))))
    print(kot)

    '''
    x_0 = np.array([1, 0]) #enotski vektor
    v_a = [A[0] - B[0], A[1] - B[1]]
    v_b = [B[0] - x_0[0], B[0] - x_0[1]]

    sk_pr = np.abs(v_a[0] * x_0[0] + v_a[1] + x_0[1])
    absA = np.sqrt(v_a[0]**2 + v_a[1]**2)
    absB = np.sqrt(x_0[0]**2 + x_0[1]**2)

    cos_fi = sk_pr/(absA*absB)
    cos_fi_rad = np.arccos(cos_fi)
    cos_fi_deg = np.degrees(cos_fi_rad)
    print(cos_fi_deg)
    '''

    I_rot = getParameters('affine', rot = -(180-kot), center=(354,156))
    I_rot_trans = transformImage('affine', I_grey, [1,1], iP = np.linalg.inv(I_rot), iInterp= 1, iBgr = 0)
    displayImage(I_rot_trans)

    # 3. Naloga
    sobel_x = spatialFiltering('kernel', I_rot_trans, Sb_x)
    sobel_y = spatialFiltering('kernel', I_rot_trans, Sb_y)
    Sb_A = np.sqrt(sobel_x**2 + sobel_y **2) #edge_detection
    displayImage(Sb_A)

    I_rot_tresh = tresholdImage(Sb_A, 254)
    displayImage(I_rot_tresh)


    # 4. Naloga
    getSquareCenterPoint(I_rot_tresh, 53)
    
    #https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
    plt.figure()
    plt.plot(378,180,'ro') 
    plt.imshow(I_rot_tresh, cmap="gray")
    ax = plt.gca()
    
    rect = patches.Rectangle((352, 154), 53, 53, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # velikost stranice kvadrata:
    #pogledam vrstico slike 160 (npr) in v njej izpišem vrednosti, kjer je vrednost drugačna kot 0
    # np.where(iImage[160,:] > 0)
    # to mi da array (array([  0, 352, 353, 355, 356, 402, 404, 405, 806]),)
    # tu lahko vidimo da se kvadrat začne v 352 in gre do 405, kar pomeni da je dolžina stranice
    # a = 53





