import numpy as np
from matplotlib import pyplot as plt
import cv2

def loadImage(iPath, iSize, iType):
    #### LOADING JPEGS
    oImage = cv2.imread(iPath)
    oImage = cv2.cvtColor(oImage, cv2.COLOR_BGR2RGB)
    oImage = cv2.resize(oImage, (iSize[0], iSize[1]))
    oImage = oImage.astype(iType)
    return oImage


def displayImage(iImage, iTitle, iGridX = None, iGridY = None):
    plt.figure()
    plt.title(iTitle)
    if (iGridX is None) or (iGridY is None): 
        plt.imshow(iImage, cmap='gray', vmin=0, vmax=255,aspect = 'equal')
    else:
        plt.imshow(iImage, cmap='gray', vmin=0, vmax=255, extent=[iGridX[0], iGridX[-1], iGridY[0], iGridY[-1]],aspect = 'equal')
    if(iTitle != "Amplitudni odziv po sobelovem filtriranju in izračunanih središčih" and iTitle != "Akumulator s središčom" and iTitle != "5B      Radialna transformacija mreže"):
        plt.show()

def get_blue_region ( iImage , iThreshold ) :
    blue_bright = np.zeros((iImage.shape[0], iImage.shape[1]), dtype = np.uint8)
    bright = np.zeros((iImage.shape[0], iImage.shape[1]), dtype = np.uint8)

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            if iImage[y][x][2] > iThreshold:
                blue_bright[y][x] = 1
                if iImage[y][x][0] > iThreshold:
                    if iImage[y][x][1] > iThreshold:
                        if iImage[y][x][2] > iThreshold:
                            bright[y][x] = 1
                    
    oImage = np.zeros((iImage.shape[0],iImage.shape[1]))
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            if blue_bright[y][x]==1:
                if bright[y][x] != 1:
                    oImage[y][x] = 255


    return oImage

def changeSpatialDomain( iType , iImage , iX , iY , iMode = None , iBgr = 0) :
    if iX < 0 or iY < 0:
        print("Prosim, da popravite velikost razširitve na pozitivna števila, oziroma poročilo ocenite dobro, saj sem implementiral tudi warning :)")
        return
    Y,X = iImage.shape

    if iType == "enlarge":
        if iMode is None:
            oImage = np.zeros((Y+2*iY, X+2*iX),dtype = float)
            oImage[iY: Y + iY, iX: X+ iX] = iImage
            oImage = oImage.astype(np.uint8)
        elif iMode == "constant":
            oImage = np.ones((Y+2*iY, X+2*iX),dtype = float) * iBgr
            oImage[iY: Y + iY, iX: X+ iX] = iImage
            oImage = oImage.astype(np.uint8)
        elif iMode == "extrapolation":
            oImage = np.zeros((Y+2*iY, X+2*iX),dtype = float)
            #copy
            oImage[iY: Y + iY, iX: X+ iX] = iImage
            #top middle
            oImage[0:iY, iX: X+ iX] = iImage[0, :]
            #bottom middle
            oImage[Y+iY:Y+2*iY, iX: X+ iX] = iImage[-1, :]
            #top left
            oImage[0:iY, 0:iX] = iImage[0, 0]
            #top right
            oImage[0:iY, X+iX:X+2*iX] = iImage[0, -1]
            #bottom left
            oImage[Y+iY:Y+2*iY, 0:iX] = iImage[-1, 0]
            #bottom right
            oImage[Y+iY:Y+2*iY, X+iX:X+2*iX] = iImage[-1, -1]
            #middle left
            oImage[iY: Y + iY, 0:iX] = iImage[:, 0].reshape(-1, 1)
            #middle right
            oImage[iY: Y + iY, X+iX:X+2*iX] = iImage[:, -1].reshape(-1, 1)
            oImage = oImage.astype(np.uint8)
        elif iMode == "reflection":
            oImage = np.zeros((Y+2*iY, X+2*iX),dtype = float)
            oImage[iY: Y + iY, iX: X+ iX] = np.copy(iImage)
            copy = np.copy(iImage)
            #read copy from behind on both axis
            mirroredboth = np.copy(copy)
            mirroredboth = np.flip(mirroredboth, axis = 0)
            mirroredboth = np.flip(mirroredboth, axis = 1)
            #read copy from behind on x axis
            mirroredx = np.copy(copy)
            mirroredx = np.flip(mirroredx, axis = 1)
            #read copy from behind on y axis
            mirroredy = np.copy(copy)
            mirroredy = np.flip(mirroredy, axis = 0)
    


            diffY = iY - Y
            diffX = iX - X
        
            kY = iY//Y
            kX = iX//X

            remainderY = iY-kY*Y
            remainderX = iX-kX*X
        

            if iY >= Y and iX >= X:
                oImage[iY:Y+iY, iX :X+iX] = copy
                for z in range(1,kY+1):
                    for v in range(1,kX+1):
                        #top left
                        if (z%2 == 1 and v%2 == 1):
                            oImage[iY-z*Y:iY-(z-1)*Y,iX-v*X:iX-(v-1)*X] = mirroredboth
                        else: 
                            oImage[iY-z*Y:iY-(z-1)*Y,iX-v*X:iX-(v-1)*X] = copy #gud
                        #top right
                        if (z%2 == 1 and v%2 == 1):
                            oImage[iY-z*Y:iY-(z-1)*Y,X+iX+(v-1)*X:X+iX+v*X] = mirroredboth
                        else:
                            oImage[iY-z*Y:iY-(z-1)*Y,iX+(v)*Y:(v+1)*X+iX] = copy #gud
                        #top middle
                        if (z%2 == 1):
                            oImage[iY-z*Y:iY-(z-1)*Y,iX:iX+X] = mirroredy
                        else:
                            oImage[iY-z*Y:iY-(z-1)*Y,iX:iX+X] = copy #gud
                        #bottom left
                        if (z%2 == 1 and v%2 == 1):
                            oImage[Y+iY+(z-1)*Y:Y+iY+z*Y,iX-v*X:iX-(v-1)*X] = mirroredboth
                        else:
                            oImage[iY+(z)*Y:(z+1)*Y+iY,iX-v*X:iX-(v-1)*X] = copy #gud
                        #bottom right
                        if (z%2 == 1 and v%2 == 1):
                            oImage[Y+iY+(z-1)*Y:Y+iY+z*Y,X+iX+(v-1)*X:X+iX+v*X] = mirroredboth
                        else:
                            oImage[iY+(z)*Y:(z+1)*Y+iY,iX+(v)*Y:(v+1)*X+iX] = copy #gud
                        #bottom middle
                        if (z%2 == 1):
                            oImage[iY+(z)*Y:(z+1)*Y+iY,iX:iX+X] = mirroredy
                        else:
                            oImage[iY+(z)*Y:(z+1)*Y+iY,iX:iX+X] = copy #gud
                        #middle left
                        if (v%2 == 1):
                            oImage[iY:iY+Y,iX-v*X:iX-(v-1)*X] = mirroredx
                        else:
                            oImage[iY:iY+Y,iX-v*X:iX-(v-1)*X] = copy #gud
                        #middle right
                        if (v%2 == 1):
                            oImage[iY:iY+Y,iX+(v)*Y:(v+1)*X+iX] = mirroredx
                        else:
                            oImage[iY:iY+Y,iX+(v)*Y:(v+1)*X+iX] = copy
                        #remainders
                        #top left
                        if (z%2 == 1 and v%2 == 1):
                            oImage[0:remainderY,0:remainderX] = copy[abs(Y-remainderY):,abs(X-remainderX):]
                            
                        else:
                            oImage[0:remainderY,0:remainderX] = mirroredboth[abs(Y-remainderY):,abs(X-remainderX):]
                        #top middle
                        if (z%2 == 1):
                            oImage[0:remainderY,iX:iX+X] = copy[abs(Y-remainderY):,:]
                        else:
                            oImage[0:remainderY,iX:iX+X] = mirroredy[abs(Y-remainderY):,:]
                        #top right
                        if (z%2 == 1 and v%2 == 1):
                            oImage[0:remainderY,-remainderX:] = copy[abs(Y-remainderY):,:remainderX]
                        else:
                            oImage[0:remainderY,-remainderX:] = mirroredboth[abs(Y-remainderY):,:remainderX]
                        #left middle
                        if (v%2 == 1):
                            oImage[iY:iY+Y,0:remainderX] = copy[:,abs(X-remainderX):]
                        else:
                            oImage[iY:iY+Y,0:remainderX] = mirroredx[:,abs(X-remainderX):]
                        #right middle
                        if (v%2 == 1):
                            oImage[iY:iY+Y,-remainderX:] = copy[:,:remainderX]
                        else:
                            oImage[iY:iY+Y,-remainderX:] = mirroredx[:,:remainderX]
                        #bottom left
                        if (z%2 == 1 and v%2 == 1):
                            oImage[-remainderY:,0:remainderX] = copy[:remainderY,abs(X-remainderX):]
                        else:
                            oImage[-remainderY:,0:remainderX] = mirroredboth[:remainderY,abs(X-remainderX):]
                        #bottom middle
                        if (z%2 == 1):
                            oImage[-remainderY:,iX:iX+X] = copy[:remainderY,:]
                        else:
                            oImage[-remainderY:,iX:iX+X] = mirroredy[:remainderY,:]
                        #bottom right
                        if (z%2 == 1 and v%2 == 1):
                            oImage[-remainderY:,-remainderX:] = copy[:remainderY,:remainderX]
                        else:
                            oImage[-remainderY:,-remainderX:] = mirroredboth[:remainderY,:remainderX]
                        #top left inbetween
                        if (z%2 == 1 and v%2 == 1):
                            oImage[0:remainderY,iX-v*X:iX-(v-1)*X] = mirroredx[abs(Y-remainderY):,:]
                        else:
                            oImage[0:remainderY,iX-v*X:iX-(v-1)*X] = copy[abs(Y-remainderY):,:]
                        #top right inbetween
                        if (v%2 == 1):
                            oImage[0:remainderY,iX+(v)*Y:(v+1)*X+iX] = mirroredx[abs(Y-remainderY):,:]
                        else:
                            oImage[0:remainderY,iX+(v)*Y:(v+1)*X+iX] = copy[abs(Y-remainderY):,:]
                        #bottom left inbetween
                        if (v%2 == 1):
                            oImage[-remainderY:,iX-v*X:iX-(v-1)*X] = mirroredx[:remainderY,:]
                        else:
                            oImage[-remainderY:,iX-v*X:iX-(v-1)*X] = copy[:remainderY,:]
                        #bottom right inbetween
                        if(v%2 == 1):
                            oImage[-remainderY:,iX+(v)*Y:(v+1)*X+iX] = mirroredx[:remainderY,:]
                        else:
                            oImage[-remainderY:,iX+(v)*Y:(v+1)*X+iX] = copy[:remainderY,:]
                        #left middle inbetween up
                        if (z%2 == 1):
                            oImage[iY-z*Y:iY-(z-1)*Y,0:remainderX] = mirroredy[:,abs(X-remainderX):]
                        else:
                            oImage[iY-z*Y:iY-(z-1)*Y,0:remainderX] = copy[:,abs(X-remainderX):]
                        #right middle inbetween up
                        if (z%2 == 1):
                            oImage[iY-z*Y:iY-(z-1)*Y,-remainderX:] = mirroredy[:,:remainderX]
                        else: 
                            oImage[iY-z*Y:iY-(z-1)*Y,-remainderX:] = copy[:,:remainderX]
                        #middle left inbetween down
                        if (z%2 == 1):
                            oImage[iY+(z)*Y:(z+1)*Y+iY,0:remainderX] = mirroredy[:,abs(X-remainderX):]
                        else:
                            oImage[iY+(z)*Y:(z+1)*Y+iY,0:remainderX] = copy[:,abs(X-remainderX):]
                        #middle right inbetween down
                        if (z%2 == 1):
                            oImage[iY+(z)*Y:(z+1)*Y+iY,-remainderX:] = mirroredy[:,:remainderX]
                        else:
                            oImage[iY+(z)*Y:(z+1)*Y+iY,-remainderX:] = copy[:,:remainderX]
            elif iY >= Y and iX < X:
                for z in range(0,kY+1):
                    v=0
                    #top left
                    if (z%2 == 1):
                        oImage[0:remainderY,0:remainderX] = mirroredx[abs(Y-remainderY):,abs(X-remainderX):]
                    else:
                        oImage[0:remainderY,0:remainderX] = mirroredboth[abs(Y-remainderY):,abs(X-remainderX):]
                    #top middle
                    if (z%2 == 1):
                        oImage[0:remainderY,iX:iX+X] = copy[abs(Y-remainderY):,:]
                    else:
                        oImage[0:remainderY,iX:iX+X] = mirroredy[abs(Y-remainderY):,:]
                    #top right
                    if (z%2 == 1):
                        oImage[0:remainderY,-remainderX:] = mirroredx[abs(Y-remainderY):,:remainderX]
                    else:
                        oImage[0:remainderY,-remainderX:] = mirroredboth[abs(Y-remainderY):,:remainderX]
                    #middle left
                    oImage[iY:iY+Y,0:remainderX] = mirroredx[:,abs(X-remainderX):]
                    
                    
                    #middle right
                    oImage[iY:iY+Y,-remainderX:] = mirroredx[:,:remainderX]
                    #bottom left
                    if (z%2 == 1):
                        oImage[-remainderY:,0:remainderX] = mirroredx[:remainderY,abs(X-remainderX):]
                    else:
                        oImage[-remainderY:,0:remainderX] = mirroredboth[:remainderY,abs(X-remainderX):]
                    #bottom middle
                    if (z%2 == 1):
                        oImage[-remainderY:,iX:iX+X] = copy[:remainderY,:]
                    else:
                        oImage[-remainderY:,iX:iX+X] = mirroredy[:remainderY,:]
                    #bottom right
                    if (z%2 == 1):
                        oImage[-remainderY:,-remainderX:] = mirroredx[:remainderY,:remainderX]
                    else:
                        oImage[-remainderY:,-remainderX:] = mirroredboth[:remainderY,:remainderX]
                    #inbetween left top
                    if (z%2 == 1):
                        oImage[iY-z*Y:iY-(z-1)*Y,0:remainderX] = mirroredboth[:,abs(X-remainderX):]
                    else:
                        oImage[iY-z*Y:iY-(z-1)*Y,0:remainderX] = mirroredx[:,abs(X-remainderX):]
                    #inbetween right top
                    if (z%2 == 1):
                        oImage[iY-z*Y:iY-(z-1)*Y,-remainderX:] = mirroredboth[:,:remainderX]
                    else:
                        oImage[iY-z*Y:iY-(z-1)*Y,-remainderX:] = mirroredx[:,:remainderX]
                    #inbetween left bottom
                    if (z%2 == 1):
                        oImage[iY+(z)*Y:(z+1)*Y+iY,0:remainderX] = mirroredboth[:,abs(X-remainderX):]
                    else:
                        oImage[iY+(z)*Y:(z+1)*Y+iY,0:remainderX] = mirroredx[:,abs(X-remainderX):]
                    #inbetween right bottom
                    if (z%2 == 1):
                        oImage[iY+(z)*Y:(z+1)*Y+iY,-remainderX:] = mirroredboth[:,:remainderX]
                    else:
                        oImage[iY+(z)*Y:(z+1)*Y+iY,-remainderX:] = mirroredx[:,:remainderX]
                    #middle top middle
                    if (z%2 == 1):
                        oImage[iY-z*Y:iY-(z-1)*Y,iX:iX+X] = mirroredy
                    else:
                        oImage[iY-z*Y:iY-(z-1)*Y,iX:iX+X] = copy #gud
                    #middle bottom middle
                    if (z%2 == 1):
                        oImage[iY+(z)*Y:(z+1)*Y+iY,iX:iX+X] = mirroredy
                    else:
                        oImage[iY+(z)*Y:(z+1)*Y+iY,iX:iX+X] = copy

                        

                        

                
                

                    
                    


            elif iY < Y and iX >= X:
                oImage[iY:Y+iY, iX :X+iX] = copy
                
                for v in range(kX+1):
                    z = 0
                    #top left
                    if (v%2 == 1):
                        oImage[0:remainderY,0:remainderX] = mirroredy[abs(Y-remainderY):,abs(X-remainderX):]
                    else:
                        oImage[0:remainderY,0:remainderX] = mirroredboth[abs(Y-remainderY):,abs(X-remainderX):]
                    #top middle
                    oImage[0:remainderY,iX:iX+X] = mirroredy[abs(Y-remainderY):,:]
                    #top right
                    if (v%2 == 1):
                        oImage[0:remainderY,-remainderX:] = mirroredy[abs(Y-remainderY):,:remainderX]
                    else:
                        oImage[0:remainderY,-remainderX:] = mirroredboth[abs(Y-remainderY):,:remainderX]
                    #middle left
                    if (v%2 == 1):
                        oImage[iY:iY+Y,0:remainderX] = copy[:,abs(X-remainderX):]
                    else:
                        oImage[iY:iY+Y,0:remainderX] = mirroredx[:,abs(X-remainderX):]
                    #middle right
                    if (v%2 == 1):
                        oImage[iY:iY+Y,-remainderX:] = copy[:,:remainderX]
                    else:
                        oImage[iY:iY+Y,-remainderX:] = mirroredx[:,:remainderX]
                    #bottom left
                    if (v%2 == 1):
                        oImage[-remainderY:,0:remainderX] = mirroredy[:remainderY,abs(X-remainderX):]
                    else:
                        oImage[-remainderY:,0:remainderX] = mirroredboth[:remainderY,abs(X-remainderX):]
                    #bottom middle
                    oImage[-remainderY:,iX:iX+X] = mirroredy[:remainderY,:]
                    #bottom right
                    if (v%2 == 1):
                        oImage[-remainderY:,-remainderX:] = mirroredy[:remainderY,:remainderX]
                    else:
                        oImage[-remainderY:,-remainderX:] = mirroredboth[:remainderY,:remainderX]
                    #inbetween right top
                    if (v%2 == 1):
                        oImage[0:remainderY,iX+v*X:iX+(v+1)*X] = mirroredboth[abs(Y-remainderY):,:]
                    else:
                        oImage[0:remainderY,iX+v*X:iX+(v+1)*X] = mirroredy[abs(Y-remainderY):,:]
                    #inbetween left top
                    if (v%2 == 1):
                        oImage[0:remainderY,iX-v*X:iX-(v-1)*X] = mirroredboth[abs(Y-remainderY):,:]
                    else:
                        oImage[0:remainderY,iX-v*X:iX-(v-1)*X] = mirroredy[abs(Y-remainderY):,:]
                    #inbetween right bottom
                    if (v%2 == 1):
                        oImage[-remainderY:,iX+v*X:iX+(v+1)*X] = mirroredboth[:remainderY,:]
                    else:
                        oImage[-remainderY:,iX+v*X:iX+(v+1)*X] = mirroredy[:remainderY,:]
                    #inbetween left bottom
                    if (v%2 == 1):
                        oImage[-remainderY:,iX-v*X:iX-(v-1)*X] = mirroredboth[:remainderY,:]
                    else:
                        oImage[-remainderY:,iX-v*X:iX-(v-1)*X] = mirroredy[:remainderY,:]
                    #inbetween left middle
                    if (v%2 == 1):
                        oImage[iY:iY+Y,iX-v*X:iX-(v-1)*X] = mirroredx
                    else:
                        oImage[iY:iY+Y,iX-v*X:iX-(v-1)*X] = copy
                    #inbetween right middle
                    if (v%2 == 1):
                        oImage[iY:iY+Y,iX+v*X:iX+(v+1)*X] = mirroredx
                    else:
                        oImage[iY:iY+Y,iX+v*X:iX+(v+1)*X] = copy


                

            elif iY < Y and iX < X:
                oImage[iY:Y+iY, iX :X+iX] = copy
                #remainders
                #top left
                oImage[0:remainderY,0:remainderX] = mirroredboth[abs(Y-remainderY):,abs(X-remainderX):]
                #top middle
                oImage[0:remainderY,iX:iX+X] = mirroredy[abs(Y-remainderY):,:]
                #top right
                oImage[0:remainderY,-remainderX:] = mirroredboth[abs(Y-remainderY):,:remainderX]
                #middle left
                oImage[iY:iY+Y,0:remainderX] = mirroredx[:,abs(X-remainderX):]
                #middle right
                oImage[iY:iY+Y,-remainderX:] = mirroredx[:,:remainderX]
                #bottom left
                oImage[-remainderY:,0:remainderX] = mirroredboth[:remainderY,abs(X-remainderX):]
                #bottom middle
                oImage[-remainderY:,iX:iX+X] = mirroredy[:remainderY,:]
                #bottom right
                oImage[-remainderY:,-remainderX:] = mirroredboth[:remainderY,:remainderX]

                    
            

                oImage = oImage.astype(np.uint8)
            
        # turn this into a for loop that makes it repeat itself so that it fills the entire canvas
        elif iMode == "period":
            oImage = np.zeros((Y+2*iY, X+2*iX),dtype = float)
            oImage[iY: Y + iY, iX: X+ iX] = np.copy(iImage)
            copy = np.copy(iImage)

            diffY = iY - Y
            diffX = iX - X
        
            kY = iY//Y
            kX = iX//X

            remainderY = iY-kY*Y
            remainderX = iX-kX*X
            

            if iY >= Y and iX >= X:
                oImage[iY:Y+iY, iX :X+iX] = copy
                for z in range(1,kY+1):
                    for v in range(1,kX+1):
                        #top left
                        oImage[iY-z*Y:iY-(z-1)*Y,iX-v*X:iX-(v-1)*X] = copy #gud
                        #top right
                        oImage[iY-z*Y:iY-(z-1)*Y,iX+(v)*Y:(v+1)*X+iX] = copy #gud
                        #top middle
                        oImage[iY-z*Y:iY-(z-1)*Y,iX:iX+X] = copy #gud
                        #bottom left
                        oImage[iY+(z)*Y:(z+1)*Y+iY,iX-v*X:iX-(v-1)*X] = copy #gud
                        #bottom right
                        oImage[iY+(z)*Y:(z+1)*Y+iY,iX+(v)*Y:(v+1)*X+iX] = copy #gud
                        #bottom middle
                        oImage[iY+(z)*Y:(z+1)*Y+iY,iX:iX+X] = copy #gud
                        #middle left
                        oImage[iY:iY+Y,iX-v*X:iX-(v-1)*X] = copy #gud
                        #middle right
                        oImage[iY:iY+Y,iX+(v)*Y:(v+1)*X+iX] = copy #gud
                        #remainders
                        #top left
                        oImage[0:remainderY,0:remainderX] = copy[abs(Y-remainderY):,abs(X-remainderX):]
                        #top middle
                        oImage[0:remainderY,iX:iX+X] = copy[abs(Y-remainderY):,:]
                        #top right
                        oImage[0:remainderY,-remainderX:] = copy[abs(Y-remainderY):,:remainderX]
                        #left middle
                        oImage[iY:iY+Y,0:remainderX] = copy[:,abs(X-remainderX):]
                        #right middle
                        oImage[iY:iY+Y,-remainderX:] = copy[:,:remainderX]
                        #bottom left
                        oImage[-remainderY:,0:remainderX] = copy[:remainderY,abs(X-remainderX):]
                        #bottom middle
                        oImage[-remainderY:,iX:iX+X] = copy[:remainderY,:]
                        #bottom right
                        oImage[-remainderY:,-remainderX:] = copy[:remainderY,:remainderX]
                        #top left inbetween
                        oImage[0:remainderY,iX-v*X:iX-(v-1)*X] = copy[abs(Y-remainderY):,:]
                        #top right inbetween
                        oImage[0:remainderY,iX+(v)*Y:(v+1)*X+iX] = copy[abs(Y-remainderY):,:]
                        #bottom left inbetween
                        oImage[-remainderY:,iX-v*X:iX-(v-1)*X] = copy[:remainderY,:]
                        #bottom right inbetween
                        oImage[-remainderY:,iX+(v)*Y:(v+1)*X+iX] = copy[:remainderY,:]
                        #left middle inbetween up
                        oImage[iY-z*Y:iY-(z-1)*Y,0:remainderX] = copy[:,abs(X-remainderX):]
                        #right middle inbetween up
                        oImage[iY-z*Y:iY-(z-1)*Y,-remainderX:] = copy[:,:remainderX]
                        #middle left inbetween down
                        oImage[iY+(z)*Y:(z+1)*Y+iY,0:remainderX] = copy[:,abs(X-remainderX):]
                        #middle right inbetween down
                        oImage[iY+(z)*Y:(z+1)*Y+iY,-remainderX:] = copy[:,:remainderX]
            elif iY >= Y and iX < X:
                oImage[iY:Y+iY, iX :X+iX] = copy
                #remainders
                #top middle
                oImage[0:remainderY,iX:iX+X] = copy[abs(Y-remainderY):,:]
                #bottom middle
                oImage[-remainderY:,iX:iX+X] = copy[:remainderY,:]
                #top left
                oImage[0:remainderY,0:remainderX] = copy[abs(Y-remainderY):,abs(X-remainderX):]
                #top right
                oImage[0:remainderY,-remainderX:] = copy[abs(Y-remainderY):,:remainderX]
                #bottom left
                oImage[-remainderY:,0:remainderX] = copy[:remainderY,abs(X-remainderX):]
                #bottom right
                oImage[-remainderY:,-remainderX:] = copy[:remainderY,:remainderX]
                #left middle
                oImage[iY:iY+Y,0:remainderX] = copy[:,abs(X-remainderX):]
                #right middle
                oImage[iY:iY+Y,-remainderX:] = copy[:,:remainderX]
                for z in range(1,kY+1):
                    #top middle
                    oImage[iY-z*Y:iY-(z-1)*Y,iX:iX+X] = copy
                    #bottom middle
                    oImage[iY+(z)*Y:(z+1)*Y+iY,iX:iX+X] = copy
                    #top left inbetween
                    oImage[iY-z*Y:iY-(z-1)*Y,0:remainderX] = copy[:,abs(X-remainderX):]
                    #top right inbetween
                    oImage[iY-z*Y:iY-(z-1)*Y,-remainderX:] = copy[:,:remainderX]
                    #bottom left inbetween
                    oImage[iY+(z)*Y:(z+1)*Y+iY,0:remainderX] = copy[:,abs(X-remainderX):]
                    #bottom right inbetween
                    oImage[iY+(z)*Y:(z+1)*Y+iY,-remainderX:] = copy[:,:remainderX]

                    
                    


            elif iY < Y and iX >= X:
                oImage[iY:Y+iY, iX :X+iX] = copy
                
                #remainders
                #top left
                oImage[0:remainderY,0:remainderX] = copy[abs(Y-remainderY):,abs(X-remainderX):]
                #top middle
                oImage[0:remainderY,iX:iX+X] = copy[abs(Y-remainderY):,:]
                #top right  
                oImage[0:remainderY,-remainderX:] = copy[abs(Y-remainderY):,:remainderX]
                #bottom left
                oImage[-remainderY:,0:remainderX] = copy[:remainderY,abs(X-remainderX):]
                #bottom middle
                oImage[-remainderY:,iX:iX+X] = copy[:remainderY,:]
                #bottom right
                oImage[-remainderY:,-remainderX:] = copy[:remainderY,:remainderX]
                #left middle
                oImage[iY:iY+Y,0:remainderX] = copy[:,abs(X-remainderX):]
                #right middle
                oImage[iY:iY+Y,-remainderX:] = copy[:,:remainderX]
                for v in range(1,kX+1):
                
                    #middle left
                    oImage[iY:iY+Y,iX-v*X:iX-(v-1)*X] = copy
                    #middle right
                    oImage[iY:iY+Y,iX+(v)*Y:(v+1)*X+iX] = copy
                    #top left inbetween
                    oImage[0:remainderY,iX-v*X:iX-(v-1)*X] = copy[abs(Y-remainderY):,:]
                    #top right inbetween
                    oImage[0:remainderY,iX+(v)*Y:(v+1)*X+iX] = copy[abs(Y-remainderY):,:]
                    #bottom left inbetween
                    oImage[-remainderY:,iX-v*X:iX-(v-1)*X] = copy[:remainderY,:]
                    #bottom right inbetween
                    oImage[-remainderY:,iX+(v)*Y:(v+1)*X+iX] = copy[:remainderY,:]

            elif iY < Y and iX < X:
                oImage[iY:Y+iY, iX :X+iX] = copy
                #remainders
                #top left
                oImage[0:remainderY,0:remainderX] = copy[abs(Y-remainderY):,abs(X-remainderX):]
                #top middle
                oImage[0:remainderY,iX:iX+X] = copy[abs(Y-remainderY):,:]
                #top right
                oImage[0:remainderY,-remainderX:] = copy[abs(Y-remainderY):,:remainderX]
                #bottom left
                oImage[-remainderY:,0:remainderX] = copy[:remainderY,abs(X-remainderX):]
                #bottom middle
                oImage[-remainderY:,iX:iX+X] = copy[:remainderY,:]
                #bottom right
                oImage[-remainderY:,-remainderX:] = copy[:remainderY,:remainderX]
                #left middle
                oImage[iY:iY+Y,0:remainderX] = copy[:,abs(X-remainderX):]
                #right middle
                oImage[iY:iY+Y,-remainderX:] = copy[:,:remainderX]
                for v in range(1,kX+1):
                    #middle left
                    oImage[iY:iY+Y,iX-v*X:iX-(v-1)*X] = copy
                    #middle right
                    oImage[iY:iY+Y,iX+(v)*Y:(v+1)*X+iX] = copy
                    #top left inbetween
                    oImage[0:remainderY,iX-v*X:iX-(v-1)*X] = copy[abs(Y-remainderY):,:]
                    #top right inbetween
                    oImage[0:remainderY,iX+(v)*Y:(v+1)*X+iX] = copy[abs(Y-remainderY):,:]
                    #bottom left inbetween
                    oImage[-remainderY:,iX-v*X:iX-(v-1)*X] = copy[:remainderY,:]
                    #bottom right inbetween
                    oImage[-remainderY:,iX+(v)*Y:(v+1)*X+iX] = copy[:remainderY,:]
                for z in range(1,kY+1):
                    #middle top
                    oImage[iY-z*Y:iY-(z-1)*Y,iX:iX+X] = copy
                    #middle bottom
                    oImage[iY+(z)*Y:(z+1)*Y+iY,iX:iX+X] = copy
                    #top left inbetween
                    oImage[iY-z*Y:iY-(z-1)*Y,0:remainderX] = copy[:,abs(X-remainderX):]
                    #top right inbetween
                    oImage[iY-z*Y:iY-(z-1)*Y,-remainderX:] = copy[:,:remainderX]
                    #bottom left inbetween
                    oImage[iY+(z)*Y:(z+1)*Y+iY,0:remainderX] = copy[:,abs(X-remainderX):]
                    #bottom right inbetween
                    oImage[iY+(z)*Y:(z+1)*Y+iY,-remainderX:] = copy[:,:remainderX]


            

                oImage = oImage.astype(np.uint8)


            
            

    elif iType == "reduce":
        oImage = np.copy(iImage[iY:Y-iY, iX:X-iX])

    
        
    return oImage




def spatialFiltering(iType, iImage, iFilter, iStatFunc = None, iMorphOp = None):
    N, M = iFilter.shape
    n = int((N-1)/2)
    m = int((M-1)/2)
    
    iImage = changeSpatialDomain("enlarge", iImage, iX = m, iY = n)

    Y, X = iImage.shape
    oImage = np.zeros(iImage.shape)

    if iType == "kernel":
        for y in range(n, Y-n):     #prva možna ne sme biti 0,0 ampak moramo premakniti za polovico filtra
            for x in range(m, X-m):
                patch = iImage[y-n:y+n+1, x-m:x+m+1]  #da vključimo še zadnji element moramo dodati 1 cause python
                oImage[y,x] = np.sum(patch * iFilter)  ##izločiš patch, ga pomnožiš z filterjem in sešteješ
        
    elif iType == "statistical":
        for y in range(n, Y-n):
            for x in range(m, X-m):
                patch = iImage[y-n:y+n+1, x-m:x+m+1]  #da vključimo še zadnji element moramo dodati 1 cause python
                oImage[y,x] = iStatFunc(patch)
    
    elif iType == "morphological":
        for y in range(n, Y-n):
            for x in range(m, X-m):
                patch = iImage[y-n:y+n+1, x-m:x+m+1]  #da vključimo še zadnji element moramo dodati 1 cause python
                R = patch[iFilter != 0] #izpiše tiste elemente ki niso 0
                if iMorphOp == "dilation":
                    oImage[y,x] = np.max(R)
                elif iMorphOp == "erosion":
                    oImage[y,x] = np.min(R)
    oImage = changeSpatialDomain("reduce", oImage, iX = m, iY = n)
    return oImage


def compute_distances( iImage , iMask = None ):
    if iMask is None :
        distanceImg = np.zeros_like(iImage)
        for y in range(iImage.shape[0]):
            for x in range(iImage.shape[1]):
                distance = np.zeros(2720)
                for i in range(distance):
                    distance[i] = np.sqrt((edge_coordinates[i][0]-x)**2+(edge_coordinates[i][1]-y)**2)
                distanceImg[y][x] = np.amin(distance)
        
        distanceImg = distanceImg/distanceImg.max() *255
        oImage = distanceImg.copy()
    
    else:
        distanceImg = np.zeros_like(iImage)
        for y in range(iImage.shape[0]):
            for x in range(iImage.shape[1]):
                if iMask[y][x]!= 0:
                    distance = np.zeros(2720)
                    for i in range(distance.shape[0]):
                        distance[i] = np.sqrt((edge_coordinates[i][0]-x)**2+(edge_coordinates[i][1]-y)**2)
                    distanceImg[y][x] = np.amin(distance)
        
        distanceImg = distanceImg/distanceImg.max() *255
        oImage = distanceImg.copy()

    
    return oImage


def find_edge_coordinates( iImage ) :
    oEdges = np.empty((0, 2), dtype = np.uint8)
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            if iImage[y,x] != 0:
                oEdges = np.append(oEdges, np.array([[x,y]]), axis=0)
    return oEdges


if __name__ == "__main__":
    I = loadImage(r"zagovor1\data\bled-lake-decimated-uint8.jpeg", [693, 340], np.uint8)
    displayImage(I, "bled-lake-decimated-uint8.jpeg")

    Blue = get_blue_region(I, 235)
    displayImage(Blue, "Modra maska")

    SE =  np.array([
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]
        ])
    SE2 = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])

    SE3 = np.array([
            [0,0,0,1,0,0,0],
            [0,0,1,1,1,0,0],
            [0,1,1,1,1,1,0],
            [1,1,1,1,1,1,1],
            [0,1,1,1,1,1,0],
            [0,0,1,1,1,0,0],
            [0,0,0,1,0,0,0],
        ])

    BlueErosion = spatialFiltering("morphological", Blue, iFilter = SE3, iMorphOp = "erosion")
    lake_mask = spatialFiltering("morphological", BlueErosion, iFilter = SE3, iMorphOp = "dilation")
    displayImage(lake_mask, "Popravljen filter")


    SX = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]])

    SY = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]])

    sxI = spatialFiltering("kernel", lake_mask, SX)
    displayImage(sxI, "Slika po Sobelovem filtriranju po x osi")

    syI = spatialFiltering("kernel", lake_mask, SY)
    displayImage(syI, "Slika po Sobelovem filtriranju po y osi")

    sobelI = np.sqrt(sxI**2 + syI**2)
    sobelI = sobelI-sobelI.min()
    lake_edge_mask = sobelI/sobelI.max() * 255

    displayImage(lake_edge_mask, "Amplitudni odziv po sobelovem filtriranju")

    edge_coordinates=find_edge_coordinates(lake_edge_mask)

    distance_calculated = compute_distances(lake_edge_mask, lake_mask)
    displayImage(distance_calculated, "Po kalkuliranih razdaljah")

  


    