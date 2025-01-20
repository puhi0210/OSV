
import numpy as np
import matplotlib.pyplot as plt

import os,sys
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from LabVaje.OSV_lib import loadImage, displayImage, computeHistogram, displayHistogram

# NALOGA 1

iSize = (484, 699)
I = loadImage("LabVaje/IZPIT/marilyn-monroe-484x699-08bit.raw", iSize, np.uint8)
displayImage(I, "Originalna slika")
plt.show()

hist, prob, CDF, levels = computeHistogram(I)
displayHistogram(hist, levels, "histogram")

def equalize_image ( iImage ) :
    _, _, CDF, _ = computeHistogram(iImage)

    nBits = int(np.log2(iImage.max()))+1

    max_intensity = 2 ** nBits + 1

    oImage = np.zeros_like(iImage)

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            old_intensity = iImage[y,x]
            new_intensity = np.floor(CDF[old_intensity] * max_intensity)
            if new_intensity < 0:
                new_intensity = 0
            elif new_intensity > 255:
                new_intensity = 255
            oImage[y,x] = new_intensity


    return oImage


image_equalized = equalize_image ( I )
displayImage(image_equalized, "slika z izravnanim histogramom")
plt.show()

hist, prob, CDF, levels = computeHistogram(image_equalized)
displayHistogram(hist, levels, "histogram")                 


# NALOGA 2

def draw_circle ( canvas , center , radius , color ) :
    for x in range(center[0] - radius, center[0] + radius + 1):
        for y in range(center[1] - radius, center[1] + radius + 1):
            if x >= 0 and y >= 0 and x < canvas.shape[1] and y < canvas.shape[0]:
                if np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius:
                    canvas[y, x] = color
    return canvas

canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
canvas = draw_circle(canvas, (50, 50), 25, 0)

plt.imshow(canvas)
plt.title("Canvas with Circle")
plt.show()

# NALOGA 3
def create_pop_art(iImage, max_dot_radius, background_color, dot_colors):
    canvas = np.ones((*iImage.shape, 3), dtype=np.uint8) * background_color
    step = 2 * max_dot_radius

    for y in range(0, iImage.shape[0], step):
        for x in range(0, iImage.shape[1], step):
            brightness = iImage[y, x] / 255
            dot_radius = int(max_dot_radius * (1 - brightness))
            color = dot_colors[0]
            canvas = draw_circle(canvas, (x, y), dot_radius, color)
    return canvas

dot_colors = [0]
pop_art_image = create_pop_art(image_equalized, 5, 255, dot_colors)

plt.imshow(pop_art_image)
plt.title("Pop Art Image")
plt.show()

# NALOGA 4

color_choices_rgb = [
    {"background": (255, 255, 255), "dots": [(0, 0, 0), (255, 0, 0), (0, 255, 0)]},
    {"background": (255, 255, 255), "dots": [(255, 0, 0), (0, 0, 255), (255, 255, 0)]},
    {"background": (255, 255, 0), "dots": [(255, 0, 0), (0, 0, 255), (255, 0, 255)]},
    {"background": (255, 192, 203), "dots": [(75, 0, 130), (238, 130, 238), (147, 112, 219)]},
]