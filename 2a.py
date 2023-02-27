from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from matplotlib import pyplot as plt
import math
import os

sp = 1

def in_bounds(i,j,image):
    return i in range(len(image)) and j in range(len(image[0]))

def find_same(x_pix,y_pix, t_im,f_im):
    same = [x_pix,y_pix]
    #print(x_pix, y_pix)
    mn = t_im[x_pix][y_pix] + 254
    for i in range(y_pix - sp, y_pix + sp):
        for j in range(x_pix - sp, x_pix + sp):
           # mn = f_im[x_pix][y_pix]
            if in_bounds(i,j, t_im):
                new = new_min(x_pix,y_pix, i, j, mn, t_im ,f_im)
                if new < mn:
                    same = [i,j]
                    mn = new
    return same

def new_min(x_pix,y_pix, i, j, mn, t_im ,f_im):
    new = abs(t_im[x_pix][y_pix] - f_im[i][j])
    if new < mn:
        return new
    return mn




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
new_to = Image.open(toName)

to_values = np.asarray(to_image,dtype='int64').copy()
from_values = np.asarray(from_image,dtype='int64').copy()
gray_from_values = np.asarray(gray_from_image,dtype='int64').copy()
new_values = np.asarray(new_to).copy()


for i in range(len(to_values)):
    for j in range(len(to_values[0])):
        x,y = find_same(i,j,to_values,gray_from_values)
        #print(x,y, type(to_values[i][j]),type(from_values[x][y]))
        new_values[i][j] = from_values[x][y]

#
# plt.imshow(to_image)
# plt.show()


new_image = Image.fromarray(new_values)
plt.imshow(new_image)
plt.show()




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
