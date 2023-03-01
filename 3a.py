from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from matplotlib import pyplot as plt
import math
import os
import time



sp = 5

def in_bounds(i,j,image):
    return i in range(len(image)) and j in range(len(image[0]))

def find_same(x_pix,y_pix, t_im,f_im, win):
    same = [x_pix,y_pix]
    #print(x_pix, y_pix)
    mn = t_im[x_pix][y_pix] + 254
    for i in range(win):
        frame = get_pixels_in_frame(f_im, x_pix, y_pix, i)
        for f in frame:
            if in_bounds(f[0],f[1], t_im):
                new = new_min(x_pix,y_pix, f[0],f[1], mn, t_im ,f_im)
                if new < mn:
                    same = f
                    mn = new
    #
    # for i in range(y_pix - win, y_pix + win):
    #     for j in range(x_pix - win, x_pix + win):
    #        # mn = f_im[x_pix][y_pix]
    #         if in_bounds(i,j, t_im):
    #             new = new_min(x_pix,y_pix, i, j, mn, t_im ,f_im)
    #             if new < mn:
    #                 same = [i,j]
    #                 mn = new
    return same

def similarity(x_pix,y_pix, i, j, t_im ,f_im):
    tx = int(t_im[x_pix][y_pix])
    fx = int(f_im[i][j])
    #ret = abs(t_im[x_pix][y_pix] - f_im[i][j])
    ret = abs(tx - fx)
    return ret

def distance(x_pix,y_pix, i, j):
    return math.sqrt((x_pix - i) ** 2 + (y_pix - j) ** 2)

def new_min(x_pix,y_pix, i, j, mn, t_im ,f_im):
    dis = (distance(x_pix,y_pix, i, j))
    sim = (similarity(x_pix,y_pix, i, j, t_im ,f_im))
    new = abs( sim +  1.5 * dis)
    #new = abs(t_im[x_pix][y_pix] - f_im[i][j])
    if new < mn:
        return new
    return mn

def restore_colors(to_values, gray_from_values, from_values, win):
    # print("to_values: type - {}, shape - {}".format(type(to_values), to_values.shape))
    # print("gray_from_values: type - {}, shape - {}".format(type(gray_from_values), gray_from_values.shape))
    # print("from_values: type - {}, shape - {}".format(type(from_values), from_values.shape), '\n')
    # return
    new_values = np.asarray(new_to).copy()
    new_values.fill(0)

    for i in range(len(to_values)):
        for j in range(len(to_values[0])):
            x, y = find_same(i, j, to_values, gray_from_values, win)
            new_values[i][j] = from_values[x][y]
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


def get_pixels_in_frame(image, x, y,  rec_size):
    """
    Returns a list of all the pixels in the frame of the rectangle that surrounds
    the pixel at (x, y) in the given image.
    """
    x1, y1, x2, y2 = get_surrounding_rectangle(image, x, y, rec_size)
    pixels = []
    for i in range(y1, y2+1):
        for j in range(x1, x2+1):
            if i == y1 or i == y2 or j == x1 or j == x2:
                pixels.append([i,j])
    return pixels

def get_surrounding_rectangle(image, x, y, rec_size):
    """
    Returns the rectangle that surrounds the pixel at (x, y) in the given image.
    The rectangle is represented as a tuple of (x1, y1, x2, y2), where (x1, y1) is
    the top-left corner of the rectangle and (x2, y2) is the bottom-right corner.
    """
    width = len(image[0])
    height = len(image)
    # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
    x1 = max(0, x - rec_size)
    y1 = max(0, y - rec_size)
    x2 = min(width - rec_size, x + rec_size)
    y2 = min(height - rec_size, y + rec_size)
    return (x1, y1, x2, y2)


#persons = ['cgboyc', 'cmkirk', 'djhugh','dmwest', 'gmwate','khughe','lejnno']
persons = ['cgboyc', 'cmkirk']

# toName = 'C:/Users/holtz/OneDrive/Desktop/faces_sets/training_set/' + \
#         'cgboyc' + '.' + str(12) + '.jpg'
# fromName = 'C:/Users/holtz/OneDrive/Desktop/faces_sets/training_set/' + \
#             'cgboyc' + '.' + str(16) + '.jpg'
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


# in this stage we want to load the images
# we change them to np.array to be ready for pca
start_time = time.time()

personsGreyAsMatrix = []
names_train_set = []

# get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# construct the relative path to the training set directory
training_set_dir = os.path.join(current_dir, "faces_sets", "training_set")

for name in persons:
    for i in range(1, 20):
        # construct the relative path to the image
        image_path = os.path.join(training_set_dir, name + '.' + str(i) + '.jpg')

        # open the image
        image = Image.open(image_path).convert('L')
        data = np.asarray(image)
        data = data.flatten()
        personsGreyAsMatrix.append(data)
        names_train_set.append(name + '.' + str(i) + '.jpg')
personsGreyAsMatrix = np.asarray(personsGreyAsMatrix)


test_set = []
names_test_set = []

# construct the relative path to the training set directory
test_set_dir = os.path.join(current_dir, "faces_sets", "test_set")

for name in persons:
    # construct the relative path to the image
    image_path = os.path.join(test_set_dir, name + '.' + str(20) + '.jpg')

    # open the image
    image = Image.open(image_path).convert('L')

    data = np.asarray(image)
    data = data.flatten()
    test_set.append(data)
    names_test_set.append(name + '.' + str(20) + '.jpg')
test_set = np.asarray(test_set)

#Perform PCA
n_components = len(personsGreyAsMatrix)
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)
X_train_pca = pca.fit_transform(personsGreyAsMatrix)
X_train_1d = X_train_pca.reshape((X_train_pca.shape[0], -1))

for name in persons:
    # construct the relative path to the image
    image_path = os.path.join(test_set_dir, name + '.' + str(20) + '.jpg')

    # open the image
    image = Image.open(image_path).convert('L')

    # imageName = test_path + name + '.' + str(20) + '.jpg'
    # # Open the image form working directorys
    # image = Image.open(imageName).convert('L')
    image_array = np.asarray(image)
    data = image_array.flatten()
    data = data.reshape(1,-1)
    data_pca = pca.transform(data)
    similar = cosine_similarity(data_pca,X_train_pca)
    max_similarity_index = np.argmax(similar[0])

    #image_test = Image.open(imageName)
    image_path = os.path.join(training_set_dir, names_train_set[max_similarity_index])

    res_image = Image.open(image_path)


    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(7, 7))

    # Display the images on the subplots
    ax1[0].imshow(image, cmap='gray')
    ax1[0].axis('off')
    ax1[1].imshow(res_image.convert('L'), cmap='gray')
    ax1[1].axis('off')
    #ax3.imshow(gray_from_image, cmap='gray')
    # Set titles for the subplots
    ax1[0].set_title('original')
    ax1[1].set_title('most similar')
    # ax3.set_title('gray original')


    grey_res = res_image.convert('L')
    grey_res_val = np.asarray(grey_res)
    res_values = np.asarray(res_image)

    x = restore_colors(image_array, grey_res_val, res_values, 2)
    ax2[0].imshow(x)
    ax2[0].set_title('restore image')
    ax2[0].axis('off')
    ax2[1].axis('off')

    # Show the plot
    plt.show()
    # plots_path = r'C:\Users\lenovo\Downloads\plots/'
    # plt.savefig(plots_path + name)

print("Time taken: {:.3f}m".format((time.time() - start_time)/ 60) )