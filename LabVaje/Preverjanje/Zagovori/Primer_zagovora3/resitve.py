"""Zagovor laboratorijskih vaj - Naloga 2022-2, resitve"""
import numpy as np
import matplotlib.pyplot as plt
from vaja03.code.python.script import displayImage


# -------------------------------------------------------------
## 1. NALOGA
def standardize_image(iImage):
    """
    Funkcija, ki standardizira vsak kanal vhodne slike
    """
    iImage = iImage.astype(float)
    oImage = np.copy(iImage)

    red = iImage[:, :, 0]
    r_mean = red.mean()
    r_std = red.std()
    red -= r_mean
    red /= r_std

    green = iImage[:, :, 1]
    g_mean = green.mean()
    g_std = green.std()
    green -= g_mean
    green /= g_std

    blue = iImage[:, :, 2]
    b_mean = blue.mean()
    b_std = blue.std()
    blue -= b_mean
    blue /= b_std
    oImage[:, :, 0] = red
    oImage[:, :, 1] = green
    oImage[:, :, 2] = blue
    return oImage, r_mean, r_std, g_mean, g_std, b_mean, b_std


if __name__ == "__main__":
    im = plt.imread("zagovor_2022_3/code/data/planina-uint8.jpeg")
    img_standardized, r_mean, r_std, g_mean, g_std, b_mean, b_std = standardize_image(
        im
    )
    displayImage(img_standardized, "Standardizirana slika")


# -------------------------------------------------------------
## 2. NALOGA
def rgb2hsl(rgb):
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    V = rgb.max(-1)
    C = V - rgb.min(-1)
    L = V - C / 2
    H = np.zeros_like(V)
    H[V == R] = (((G - B) / C))[V == R]
    H[V == G] = ((2 + (B - R) / C))[V == G]
    H[V == B] = ((4 + (R - G) / C))[V == B]
    H[C == 0] = 0
    H = H * 60
    # H = H/6%1

    # S component for HSV
    S_V = np.zeros_like(V)
    S_V[V != 0] = (C / V)[V != 0]

    # S component for HSL
    S_L = np.zeros_like(L)
    msk = np.logical_or(L == 0, L == 1)
    S_L[~msk] = (C / (1 - np.abs(2 * V - C - 1)))[~msk]

    hsl = np.stack((H, S_L, L), axis=-1)
    hsv = np.stack((H, S_V, V), axis=-1)
    return hsl


if __name__ == "__main__":
    img_hsv = rgb2hsl(img_standardized)
    displayImage(img_hsv, "img_hsv")

# -------------------------------------------------------------
## 3. NALOGA
if __name__ == "__main__":
    nebo_mask = np.logical_and(img_hsv[:, :, 0] >= 100, img_hsv[:, :, 0] < 250)
    displayImage(img_hsv[:, :, 0], "Kanal H")
    displayImage(nebo_mask * 255, "Maska neba")
    img_hsv_transformed = np.copy(img_hsv)
    img_hsv_transformed[nebo_mask, 0] = img_hsv_transformed[nebo_mask, 0] / 5
    displayImage(img_hsv_transformed, "img_hsv_transformed")


# -------------------------------------------------------------
## 4. NALOGA
def hsl2rgb(hsl):
    H = hsl[..., 0]
    S_L = hsl[..., 1]
    L = hsl[..., 2]
    C = (1 - np.abs(2 * L - 1)) * S_L
    H_hat = H / 60
    X = C * (1 - np.abs(H_hat % 2 - 1))
    R1 = np.zeros_like(H)
    G1 = np.zeros_like(H)
    B1 = np.zeros_like(H)

    msk = np.logical_and(0 <= H_hat, H_hat < 1)
    R1[msk] = C[msk]
    G1[msk] = X[msk]

    msk = np.logical_and(1 <= H_hat, H_hat < 2)
    R1[msk] = X[msk]
    G1[msk] = C[msk]

    msk = np.logical_and(2 <= H_hat, H_hat < 3)
    G1[msk] = C[msk]
    B1[msk] = X[msk]

    msk = np.logical_and(3 <= H_hat, H_hat < 4)
    G1[msk] = X[msk]
    B1[msk] = C[msk]

    msk = np.logical_and(4 <= H_hat, H_hat < 5)
    R1[msk] = X[msk]
    B1[msk] = C[msk]

    msk = np.logical_and(5 <= H_hat, H_hat < 6)
    R1[msk] = C[msk]
    B1[msk] = X[msk]

    m = L - C / 2
    return np.stack((R1 + m, G1 + m, B1 + m), axis=-1)


if __name__ == "__main__":
    img_hsv_to_rgb = hsl2rgb(img_hsv)
    displayImage(img_hsv_to_rgb, "img_hsv_to_rgb")
    img_rgb_transformed = hsl2rgb(img_hsv_transformed)
    displayImage(img_rgb_transformed, "img_rgb_transformed")
