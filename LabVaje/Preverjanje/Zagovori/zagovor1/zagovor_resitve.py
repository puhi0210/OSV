import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('./')
from vaja01.skripta01 import displayImage
from vaja01.skripta01 import loadImage
from vaja07.skripta07 import spatialFiltering

def get_blue_region(iImage, iThreshold):
    # print(f"oImage.shape {oImage.shape}")

    image_blue = iImage[:,:,2] > iThreshold
    image_bright = np.logical_and(iImage[:,:,2] > iThreshold, np.logical_and(iImage[:,:,0] > iThreshold, iImage[:,:,1] > iThreshold))
    
    oImage = 255*np.array(np.logical_and(image_blue, np.logical_not(image_bright)))
    # plt.figure()
    # plt.imshow(oImage, cmap="gray")
    
    return oImage

def find_edge_coordinates(iImage):
    Y,X = iImage.shape

    oEdges = []

    for x in range(X):
        for y in range(Y):
            if iImage[y,x]:
                oEdges.append([x,y])
            
    return np.array(oEdges)

def compute_distance(iImage, iMask=None):
    oImage = np.zeros_like(iImage)

    Y, X = iImage.shape

    edges = find_edge_coordinates(iImage)

    for x in range(X):
        for y in range(Y):
            if iMask is None or iMask[y,x]:
                #calculate distance to closest edge point
                distances = ( (edges[:,0] - x)**2 + (edges[:,1] - y)**2 )**0.5
                oImage[y,x] = np.min(distances)

    # if iMask is None:
        

    return oImage


if __name__=="__main__":
    print("Zagovor:")
    bled=mpimg.imread(r"zagovor1\data\bled-lake-decimated-uint8.jpeg")

    plt.figure()
    plt.imshow(bled)

    bled_blue_channel = bled[:,:,2]

    plt.figure()
    plt.imshow(bled_blue_channel, cmap="gray")

    print(f"Bled image shape: {bled.shape}")

    lake = get_blue_region(bled, 235)

    displayImage(lake,iTitle="jezero")

    # Cleanup - get just the lake
    SE5 = np.array([ [0,0,1,0,0],
                    [0,1,1,1,0],
                    [1,1,1,1,1],
                    [0,1,1,1,0],
                    [0,0,1,0,0]])

    SE7 = np.array([[0,0,0,1,0,0,0],
                    [0,0,1,1,1,0,0],
                    [0,1,1,1,1,1,0],
                    [1,1,1,1,1,1,1],
                    [0,1,1,1,1,1,0],
                    [0,0,1,1,1,0,0],
                    [0,0,0,1,0,0,0],])

    SE3 = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]])

    lake_mask = spatialFiltering("morphological", lake, iFilter=SE7, iMorphOp="erosion")
    lake_mask = spatialFiltering("morphological", lake_mask, iFilter=SE7, iMorphOp="dilation")
    displayImage(lake_mask, "Bled erosion + dilation, SE 7x7")

    # Edge detection = dilation - mask
    dilated = spatialFiltering("morphological", lake_mask, iFilter=SE3, iMorphOp="dilation")

    lake_edge_mask = dilated - lake_mask
    displayImage(lake_edge_mask,iTitle="lake_edge_mask")

    edges = find_edge_coordinates(lake_edge_mask)
    print(f"Velikost matrike robov: {edges.shape}")

    dist = compute_distance(lake_edge_mask, lake_mask)

    # Končni prikaz
    # loc_max = [int(np.argmax(image_distances)%image_distances.shape[1]), int(np.argmax(image_distances)/image_distances.shape[0])]

    loc_max = np.unravel_index(dist.argmax(), dist.shape)
    print(f"lokacija maximuma: {loc_max[1], loc_max[0]}, oddaljenost od obale: {dist[loc_max]}")
    dist = 255*dist/np.max(dist)
    image_distances = dist + lake_edge_mask
    
    displayImage(image_distances,iTitle="image distances")
    plt.plot(loc_max[1], loc_max[0], "rx")
    # print(f"max = {np.max(image_distances)}, min = {np.min(image_distances)}")


