import os
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import sys


def pca(om, cn):
    ipca = PCA(cn).fit(om)
    img_c = ipca.transform(om)

    print img_c.shape
    print np.sum(ipca.explained_variance_ratio_)

    temp = ipca.inverse_transform(img_c)
    print temp.shape

    return temp


def compressImage(image, cn):
    redChannel = image[..., 0]
    greenChannel = image[..., 1]
    blueChannel = image[..., 2]

    cmpRed = pca(redChannel, cn)
    cmpGreen = pca(greenChannel, cn)
    cmpBlue = pca(blueChannel, cn)

    newImage = np.zeros((image.shape[0], image.shape[1], 3), 'uint8')

    newImage[..., 0] = cmpRed
    newImage[..., 1] = cmpGreen
    newImage[..., 2] = cmpBlue

    return newImage


path = 'sid.jpg'
img = mpimg.imread(path)

title = "Original Image"
plt.title(title)
plt.imshow(img)
plt.show()

weights = [100, 50, 20, 5]

for cn in weights:
    newImg = compressImage(img, cn)

    title = " Image after =  %s" % cn
    plt.title(title)
    plt.imshow(newImg)
    plt.show()

    newname = os.path.splitext(path)[0] + '_comp_' + str(cn) + '.jpg'
    mpimg.imsave(newname, newImg)
