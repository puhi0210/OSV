from pickletools import uint8
from matplotlib import pyplot as plt
import numpy as np
from Vaja_2.skripta import loadImage
from Vaja_4.skripta import displayImage3D as displayImage

I = loadImage("Zagovori/Primer_zagovora1/data/rose-366-366-08bit.raw", (366,366), np.uint8)


# 1. naloga
# izreži sliko

def getBoundaryIndices(iImage, iAxis):
    oIdx = []
    #oIdx2 = []
    if iAxis == 1:
        for i in range(iImage.shape[0]):
            for j in range(iImage.shape[1]):
                if iImage[i][j] != 255: #barva ozadja je 255
                    oIdx.append(i)
                    #print(i) #sprinta indexe kjer vrednost ni samo 255
    if iAxis == 2:
            iImage = iImage.T #obrnemo sliko
            for i in range(iImage.shape[0]):
                for j in range(iImage.shape[1]):
                    if iImage[i][j] != 255: #barva ozadja je 255
                        oIdx.append(i)
                        #print(oIdx)
                        #print(i) #sprinta indexe kjer vrednost ni samo 255
    oIdx1 = oIdx[0]
    oIdx2 = oIdx[-1]
    return oIdx1, oIdx2

'''
def expandImage(iImage):
    Y, X = iImage.shape
    N, M = 2*X, 2*Y #isto kot pri interpolate image, najprej X potem Y, poglej npr. gornjo vrstico
    oImage = np.zeros((M, N), dtype = float) # ne pozabi, v Y so Y koordinate, v X pa X koordinate. N - X, M - Y.

    dx = (X-1)/(N-1) #vhodna/izhodna
    dy = (Y-1)/(M-1)

    for n in range(N):
        for m in range(M):
            s = 255 #intenziteta
            
            #iz koordinat (0,0) gremo v (1/3,0) ipd. Množimo koordinato z stepom
            pt_x = n * dx #pt - point
            pt_y = m * dy

            px_x = round(pt_x) #px - pixel
            px_y = round(pt_y) #zaokrožimo obe vrednosti 
            
            s = iImage[px_y, px_x]

            oImage[m,n] = s
    return oImage


def expandImage(iImage):
    Y, X = iImage.shape
    N, M = 2*X, 2*Y
    #oImage = np.zeros((M,N), dtype = float)

    dx = (X-1)/(N-1) #vhodna/izhodna
    dy = (Y-1)/(M-1)

    x_manjkajo = X - iImage.shape[1]
    y_manjkajo = Y - iImage.shape[0]

    oImage = np.pad(iImage, ((0,210), (84, 84)))

    return oImage
'''


def expandImage(iImage):
    Y, X = iImage.shape
    #Y, X = 2*Y, 2*X
    Y, X = 2*Y, (int(X/2) + X)
    oImage = np.pad(iImage, ((0,int(Y/2)), (int(X/2), int(X/2))), mode='constant', constant_values=255)
    return oImage


# 4. naloga:
# zadnja alineja pove da je treba sliko utežiti (spomni se filtrov)
# ideja je zarotirat sliko za iAngle in jo potem naštancat po sliki
# na koncu utežiš da se vse vidi

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

I_param_c = getParameters('affine', rot = -30)
I_affine_c = transformImage('affine', I2, px_dim, iP = np.linalg.inv(I_param_c), iInterp= 1, iBgr = bgr)

def createRotatedPattern(iImage, iAngle):
    X, Y = iImage.shape
    oImage = np.zeros((Y, X)) 
    imgCenter = (np.array([X, Y]) - 1) / 2 
    preslikave = int(360/(iAngle)) # število preslikav
    print("število preslikav je: ", preslikave)
    rotacije = np.arange(0, 360, preslikave)
    print(rotacije)

    for i in range(preslikave):
        I_rot_a = getParameters('affine', rot = iAngle*i, center=getCenter(iImage, [1,1]))
        I_rot_b = transformImage('affine', iImage, [1,1], iP = np.linalg.inv(I_rot_a), iInterp = 1, iBgr = 255)
        #displayImage(I_rot_b)
        oImage = oImage + I_rot_b

    # NORMALIZACIJA
    oImage -= oImage.min()
    oImage /= oImage.max()
    oImage *= 255

    return oImage



if __name__ == "__main__":
    # 1. naloga
    #print(I)
    idx_1 = []
    idx_1 = getBoundaryIndices(I, 1)
    print("Omejitev v y smeri: ", idx_1)

    idx_2 = []
    idx_2 = getBoundaryIndices(I, 2)
    print("Omejitev v x smeri: ", idx_2)

    I_rez = (I[34:244, 24:192])
    displayImage(I_rez)

    # 2. naloga
    # tisti koti fore, zgleda ok
    S = I[244] #cela vrstica kjer je točka S
    #S = S[S != 255]
    #print(S)
    # steblo se začne v S[81]
    # sredina je v sivinski vrednosti 213, ki se nahaja na 
    #S_mid = np.where(S == 213)
    S = I[244][111]

    # druge točke, na četrtini višine vrtnice
    # območje vrtnice v y smeri je od 34 - 244
    # območje vrtnice v x smeri je od 24 - 192
    # višina vrtnice je 244 - 35
    # širina vrtnice je 192 - 25
    # točka L in D se nahajata v 53 + 35 (88. vrstica celotne slike)
    L = I[88]
    #L = np.where(L==214)
    L = L[94]
    #print(L)

    D = I[88]
    np.flip(D)
    # D = np.where(D==253) = 151
    D = D[151]

    # razdalja od S do L
    # NE POZABI -1 KER PYTHON ZAČNE V 0
    # LAHKO SE REŠI Z np.linalg https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
    r_L = np.array((88, 94))
    r_D = np.array((88, 151))
    r_S = np.array((244,111))

    v_LS = [r_L[0] - r_S[0], r_L[1] - r_S[1]]
    v_SD = [r_S[0] - r_D[0], r_S[1] - r_D[1]]

    skalarni_produkt = np.abs(v_LS[0]* v_SD[0] + v_LS[1] * v_SD[1])
    absL = np.sqrt(v_LS[0]**2 + v_LS[1]**2)
    absD = np.sqrt(v_SD[0]**2 + v_SD[1]**2)

    cos_fi = skalarni_produkt/(absL*absD)
    cos_fi_rad = np.arccos(cos_fi)
    cos_fi_deg = np.degrees(cos_fi_rad)
    print("Kot med točkama L in D, z izhodiščem v S, znaša: ", cos_fi_deg, "stopinj.")

    # 3. NALOGA
    # skaliranje z interpolacijo
    # velikost slike je 2* 366 x 2* 366

    I_big = expandImage(I_rez)
    displayImage(I_big)

    # 4. NALOGA
    I_rot = createRotatedPattern(I_big, 20)
    displayImage(I_rot)

    



    





    



    
    





    
