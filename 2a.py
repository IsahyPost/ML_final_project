from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from matplotlib import pyplot as plt
import math
import os


class Pixel:

    def __init__(self, x, y=None):
        if y is None:
            self.copy_constructor(x)
        else:
            self.non_copy_constructor(x, y)

    def non_copy_constructor(self, x, y):
        self.x = x
        self.y = y

    def copy_constructor(self, pix):
        self.x = pix.x
        self.y = pix.y

    def __str__(self):
        return "({0}, {1})".format(self.x, self.y)

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx ** 2 + dy ** 2) ** 0.5


def in_bounds(pxl,image):
    return pxl.x in range(len(image)) and pxl.y in range(len(image[0]))

def find_same(pxl, t_im,f_im, win):
    same = Pixel(pxl)
    mn = t_im[pxl.x][pxl.y] + 254
    for i in range(win):
        frame = get_pixels_in_frame(f_im, pxl, i)
        for f in frame:
            if in_bounds(pxl, t_im):
                new = new_min(pxl, f, mn, t_im ,f_im)
                if new < mn:
                    same = Pixel(f)
                    mn = new
    return same

def similarity(pxl1, pxl2, t_im ,f_im):
    return abs(t_im[pxl1.x][pxl1.y] - f_im[pxl2.x][pxl2.y])

def distance(pxl1, pxl2):
    return math.sqrt((pxl1.x - pxl2.x) ** 2 + (pxl1.y - pxl2.y) ** 2)

def new_min(pxl, ijpxl, mn, t_im ,f_im):
    dis = (distance(pxl, ijpxl))
    sim = (similarity(pxl, ijpxl, t_im ,f_im))
    new = abs( sim +  1.5 * dis)
    if new < mn:
        return new
    return mn

def restore_colors(to_values, gray_from_values, from_values, win):
    new_values = np.asarray(new_to).copy()
    new_values.fill(0)

    for i in range(len(to_values)):
        for j in range(len(to_values[0])):
            same_pxl = find_same(Pixel(i,j), to_values, gray_from_values, win)
            new_values[i][j] = from_values[same_pxl.x][same_pxl.y]
    return new_values

def plot_results(new_values, org_image, gray_from_image):
    new_image = Image.fromarray(new_values)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

    # Display the images on the subplots
    # ax1.imshow(org_image, cmap='gray')
    ax1.imshow(org_image)
    ax2.imshow(new_image)
    ax3.imshow(gray_from_image, cmap='gray')
    # Set titles for the subplots
    ax1.set_title('original')
    ax2.set_title('new')
    ax3.set_title('gray original')

    # Show the plot
    plt.show()


def get_pixels_in_frame(image, pxl,  rec_size):
    """
    Returns a list of all the pixels in the frame of the rectangle that surrounds
    the pixel at (x, y) in the given image.
    """
    x1, y1, x2, y2 = get_surrounding_rectangle(image, pxl, rec_size)
    pixels = []
    for i in range(y1, y2+1):
        for j in range(x1, x2+1):
            if i == y1 or i == y2 or j == x1 or j == x2:
                pixels.append(Pixel(i,j))
    return pixels

def get_surrounding_rectangle(image, pxl, rec_size):
    """
    Returns the rectangle that surrounds the pixel at (x, y) in the given image.
    The rectangle is represented as a tuple of (x1, y1, x2, y2), where (x1, y1) is
    the top-left corner of the rectangle and (x2, y2) is the bottom-right corner.
    """
    width = len(image[0])
    height = len(image)
    # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
    x1 = max(0, pxl.x - rec_size)
    y1 = max(0, pxl.y - rec_size)
    x2 = min(width - rec_size, pxl.x + rec_size)
    y2 = min(height - rec_size, pxl.y + rec_size)
    return (x1, y1, x2, y2)




persons = ['cgboyc', 'cmkirk', 'djhugh','dmwest', 'gmwate','khughe','lejnno']



toName = 'cgboyc' + '.' + str(12) + '.jpg'
fromName = 'cgboyc' + '.' + str(16) + '.jpg'
to_image = Image.open(toName).convert('L')
org_image = Image.open(toName)
from_image = Image.open(fromName)
gray_from_image = Image.open(fromName).convert('L')
new_to = Image.open(toName)

to_values = np.asarray(to_image,dtype='int64').copy()
from_values = np.asarray(from_image,dtype='int64').copy()
gray_from_values = np.asarray(gray_from_image,dtype='int64').copy()





new_values = np.asarray(new_to).copy()
new_values.fill(0)
nv = restore_colors(to_values, gray_from_values, from_values, 2)


plot_results(nv, org_image, gray_from_image)



# max sim = 221
# min sim = 0
# max dis = 260.3113520382851
# min dis = 0.0
# len 12602475








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

