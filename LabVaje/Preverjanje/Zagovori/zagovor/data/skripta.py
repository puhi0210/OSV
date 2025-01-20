from Vaja01.skripta1 import loadImage
from Vaja06.vaja06 import displayImage
from Vaja07.skripta7 import spatialFiltering
from Vaja05.skripta5 import thresholdImage
import numpy as np 
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #I=loadImage('zagovor/data/bled-lake-decimated-uint8.jpeg',[693,340],np.uint8)
    
    I=plt.imread('zagovor/data/bled-lake-decimated-uint8.jpeg')
    displayImage(I,'original')

def get_blue_region(iImage , iThreshold):
    Y,X,_=iImage.shape
    oImage=np.zeros((Y,X))
    bImage=np.zeros((Y,X))
    sImage=np.zeros((Y,X))

    bImage[iImage[:,:,1]>iThreshold]=1
    bImage[iImage[:,:,1]<=iThreshold]=0

    sImage[iImage[:,:,0]>iThreshold]=1
    sImage[iImage[:,:,0]<=iThreshold]=0
    sImage[iImage[:,:,1]>iThreshold]+=1
    sImage[iImage[:,:,2]>iThreshold]+=1

    sImage[sImage>2]=2
    sImage[sImage<=2]=0

    oImage[bImage-sImage>0]=255
    oImage[bImage-sImage<=0]=0
  
    
    

    return oImage

if __name__ == "__main__":
    blue=get_blue_region(I,235)
    displayImage(blue,'')
    K=np.array([
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0]
    ])
    k=spatialFiltering('morphological',iImage=blue,iFilter=K, iMorphOp='errosion')
    lake_mask=spatialFiltering('morphological',iImage=blue,iFilter=K, iMorphOp='dilation')
    displayImage(lake_mask,'')

    K= np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    kIy=spatialFiltering('kernel',iImage=lake_mask,iFilter=K)
    
    

    K= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kIx=spatialFiltering('kernel',iImage=lake_mask,iFilter=K)    

    lake_edge_mask=np.sqrt(kIx**2+kIy**2)
    lake_edge_mask=thresholdImage(lake_edge_mask,100)
    displayImage(lake_edge_mask,'sobelov amplitudni odziv') 


def find_edge_coordinates(iImage):   

    oEdges=[]
    Y,X=iImage.shape
    for x in range (X):
        for y in range (Y):
            if iImage[y,x]==255:
                oEdges.append([x,y])
    return oEdges

if __name__ == "__main__":
    print(find_edge_coordinates(lake_edge_mask))
    print(len(find_edge_coordinates(lake_edge_mask)))

def compute_distances(iImage , iMask = None): 
    
    if iMask is None:
        iMask=np.ones(iImage.shape)

    robovi=find_edge_coordinates(iImage)
    Y,X=iImage.shape
    oImage=np.ones((Y,X))*255
    max_r=0
    for x in range (X):
        for y in range (Y):
            if iMask[y,x]:
                for i in range (len(robovi)):
                    ##za terstiranje manjsa vrednost, do vseh točk ne pride v doglednem času
                    razdalja=np.sqrt((x-robovi[i][0])**2+(y-robovi[i][1])**2)
                    if razdalja < oImage [y,x]:
                        oImage[y,x]=razdalja
                        if razdalja>max_r:
                            max_r=razdalja
                            oPoint=[x,y]
                        

    return oImage, oPoint

if __name__ == "__main__":
    razdalje, tocka= compute_distances(lake_edge_mask,255-lake_mask)
    displayImage(razdalje,'razdalje')
    print(tocka)


    fig=plt.figure()
    
    plt.gca().set_aspect('equal')   

    fig=displayImage(razdalje,'')
    plt.plot(tocka[0], tocka[1],'rx')
    plt.show()
    

    