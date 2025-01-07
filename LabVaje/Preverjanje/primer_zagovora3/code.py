

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = plt.imread("LabVaje/Preverjanje/primer_zagovora3/travnik-uint8.jpeg")
    plt.imshow(image)
    plt.show()

def normalizeImage(iImage):
    normalized_image = image.astype(float)/iImage.max()
    return normalized_image

def rgb2hsv(iImageRGB):
    ri = iImageRGB[:, :, 0]
    gi = iImageRGB[:, :, 1] 
    bi = iImageRGB[:, :, 2]
    
    vi = iImageRGB.max(axis= 2)
    ci = vi - iImageRGB.min(axis= 2)
    li = vi - (ci/2)
    
    hi = np.zeros_like(vi)
    hi[vi == ri] = 60 * (gi[vi == ri] - bi[vi == ri]) / ci[vi == ri]
    hi[vi == gi] = 60 * (2+(bi[vi == gi] - ri[vi == gi]) / ci[vi == gi])
    hi[vi == bi] = 60 * (4+(ri[vi == bi] - gi[vi == bi]) / ci[vi == bi])
    hi[ci == 0] = 0

    si = np.zeros_like(vi)
    si = ci / vi
    si[vi == 0] = 0

    hsv_image = np.stack((hi, si, vi), axis=2)

    return hsv_image

if __name__ == "__main__":
    hsv_image = rgb2hsv(normalizeImage(image))
    plt.imshow(hsv_image)
    plt.show() 

    h_slice = hsv_image[:, :, 0]
    s_slice = hsv_image[:, :, 1]
    v_slice = hsv_image[:, :, 2]

    h_slice[h_slice < 100] = h_slice[h_slice < 100] / 2

    plt.imshow(hsv_image)
    plt.show()

def hsv2rgb(iImageHSV):
    hi = iImageHSV[:, :, 0]
    si = iImageHSV[:, :, 1]
    vi = iImageHSV[:, :, 2]

    ci = si * vi
    hi_hat = hi / 60
    xi = ci * (1 - np.abs(hi_hat % 2 - 1))

    ri = np.zeros_like(hi_hat)
    gi = np.zeros_like(hi_hat)
    bi = np.zeros_like(hi_hat)

    # 0 ≤ hi_hat < 1
    ri[(0 <= hi_hat) & (hi_hat < 1)] = ci[(0 <= hi_hat) & (hi_hat < 1)]
    gi[(0 <= hi_hat) & (hi_hat < 1)] = xi[(0 <= hi_hat) & (hi_hat < 1)]
    
    # 1 ≤ hi_hat < 2
    ri[(1 <= hi_hat) & (hi_hat < 2)] = xi[(1 <= hi_hat) & (hi_hat < 2)]
    gi[(1 <= hi_hat) & (hi_hat < 2)] = ci[(1 <= hi_hat) & (hi_hat < 2)]

    # 2 ≤ hi_hat < 3
    gi[(2 <= hi_hat) & (hi_hat < 3)] = ci[(2 <= hi_hat) & (hi_hat < 3)]
    bi[(2 <= hi_hat) & (hi_hat < 3)] = xi[(2 <= hi_hat) & (hi_hat < 3)]

    # 3 ≤ hi_hat < 4
    gi[(3 <= hi_hat) & (hi_hat < 4)] = xi[(3 <= hi_hat) & (hi_hat < 4)]
    bi[(3 <= hi_hat) & (hi_hat < 4)] = ci[(3 <= hi_hat) & (hi_hat < 4)]

    # 4 ≤ hi_hat < 5
    ri[(4 <= hi_hat) & (hi_hat < 5)] = xi[(4 <= hi_hat) & (hi_hat < 5)]
    bi[(4 <= hi_hat) & (hi_hat < 5)] = ci[(4 <= hi_hat) & (hi_hat < 5)]

    # 5 ≤ hi_hat < 6
    ri[(5 <= hi_hat) & (hi_hat < 6)] = ci[(5 <= hi_hat) & (hi_hat < 6)]
    bi[(5 <= hi_hat) & (hi_hat < 6)] = xi[(5 <= hi_hat) & (hi_hat < 6)]

    mi = vi - ci

    rgb_image = np.stack([ri + mi, gi + mi, bi + mi], axis=2)

    return rgb_image

if __name__ == "__main__":
    rgb_image = hsv2rgb(hsv_image)
    plt.imshow(rgb_image)
    plt.show()