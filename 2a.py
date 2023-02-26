from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from matplotlib import pyplot as plt
import math
import os


def in_bounds(i,j,image):
    return i in range(len(image[0])) and j in range(len(image))

def new_min(x_pix,y_pix, i, j, mn, t_im ,f_im):
    print(t_im[x_pix][y_pix], f_im[i][j], sep='\n')

    new = abs(t_im[x_pix][y_pix] - f_im[i][j])
    if new < mn:
        return new
    return mn

def find_same(x_pix,y_pix, t_im,f_im):
    same = [x_pix,y_pix]
    for i in range(y_pix - 5, y_pix + 5):
        for j in range(x_pix - 5, x_pix + 5):
            mn = f_im[i][j]
            if in_bounds(i,j, to_image):
                new = new_min(i,j,mn)
                if new != mn:
                    same = [i,j]


persons = ['cgboyc', 'cmkirk', 'djhugh','dmwest', 'gmwate','khughe','lejnno']

# for name in persons:
#     imageName = name + '.' + str(20) + '.jpg'
#     # Open the image form working directorys
#     image = Image.open(imageName).convert('L')
#     plt.imshow(image, cmap='gray')
#     plt.show()

toName = 'cgboyc' + '.' + str(12) + '.jpg'
fromName = 'cgboyc' + '.' + str(16) + '.jpg'
to_image = Image.open(toName).convert('L')
from_image = Image.open(fromName)
gray_from_image = Image.open(fromName).convert('L')





im = [[1,2,3],
      [4,5,6],
      [7,8,9]]



















# pixel_values = np.asarray(image)
# pixel_values = pixel_values.copy()



#pixel_values.fill(0)
# for i in range(10):
#     for j in range(10):
#         pixel_values[i][j] = pixel_values[100][100]
# new_image = Image.fromarray(pixel_values)
# plt.imshow(new_image)
# plt.show()


#
# # Perform PCA
# pca = PCA(n_components=X.shape[0])
# X_pca = pca.fit_transform(X)
#
