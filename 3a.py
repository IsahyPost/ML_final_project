from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import math
import os


def in_bounds(i, j, image):
    return i in range(len(image)) and j in range(len(image[0]))


def find_same(x_pix, y_pix, t_im, f_im, win):
    same = [x_pix, y_pix]
    mn = t_im[x_pix][y_pix] + 254
    for i in range(win):
        frame = get_pixels_in_frame(f_im, x_pix, y_pix, i)
        for f in frame:
            if in_bounds(f[0], f[1], t_im):
                new = new_min(x_pix, y_pix, f[0], f[1], mn, t_im, f_im)
                if new < mn:
                    same = f
                    mn = new
    return same


def similarity(x_pix, y_pix, i, j, t_im, f_im):
    return abs(int(t_im[x_pix][y_pix]) - int(f_im[i][j]))

def distance(x_pix, y_pix, i, j):
    return math.sqrt((x_pix - i) ** 2 + (y_pix - j) ** 2)

def new_min(x_pix, y_pix, i, j, mn, t_im, f_im):
    dis = (distance(x_pix, y_pix, i, j))
    sim = (similarity(x_pix, y_pix, i, j, t_im, f_im))
    new = abs(sim + 1.5 * dis)
    # new = abs(t_im[x_pix][y_pix] - f_im[i][j])
    if new < mn:
        return new
    return mn

def restore_colors(to_values, gray_from_values, from_values, win, black_img):
    for i in range(len(to_values)):
        for j in range(len(to_values[0])):
            x, y = find_same(i, j, to_values, gray_from_values, win)
            black_img[i][j] = from_values[x][y]
    return black_img

def plot_results(org_image, restored_img, most_similar, name = ''):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(7, 7))

    # Display the images on the subplots
    ax1[0].imshow(org_image, cmap='gray')
    ax1[1].imshow(most_similar.convert('L'), cmap='gray')
    ax2[0].imshow(restored_img)
    ax2[1].imshow(most_similar)

    # Set titles for the subplots
    ax1[0].set_title('original')
    ax1[1].set_title('grey most similar')
    ax2[0].set_title('restore image')
    ax2[1].set_title('most similar')

    for ax in [*ax1, *ax2]:
        ax.axis('off')

    # Show the plot
    # plt.show()
    plt.savefig(r'C:\Users\lenovo\Downloads\plots\{}.png'.format(name))

def get_pixels_in_frame(image, x, y, rec_size):
    """
    Returns a list of all the pixels in the frame of the rectangle that surrounds
    the pixel at (x, y) in the given image.
    """
    x1, y1, x2, y2 = get_surrounding_rectangle(image, x, y, rec_size)
    pixels = []
    for i in range(y1, y2 + 1):
        for j in range(x1, x2 + 1):
            if i == y1 or i == y2 or j == x1 or j == x2:
                pixels.append([i, j])
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

def create_average_image(set_dir, name):
    same_people_img = []
    for i in range(1, 20):
        # construct the relative path to the image
        image_path = os.path.join(set_dir, name + '.' + str(i) + '.jpg')

        # open the image
        image = Image.open(image_path)
        data = np.asarray(image)
        # data = data.flatten()
        same_people_img.append(data)

    # Take the mean of all images in same_people_img
    return np.mean(same_people_img, axis=0)



# = ['cgboyc', 'cmkirk', 'djhugh', 'dmwest', 'gmwate', 'khughe', 'lejnno']
persons = ['cgboyc', 'cmkirk']
# persons = ['cgboyc']

# in this stage we want to load the images
# we change them to np.array to be ready for pca

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

# Perform PCA
n_components = len(personsGreyAsMatrix)
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)
X_train_pca = pca.fit_transform(personsGreyAsMatrix)
X_train_1d = X_train_pca.reshape((X_train_pca.shape[0], -1))

for name in persons:
    # construct the relative path to the image
    image_path = os.path.join(test_set_dir, name + '.' + str(20) + '.jpg')

    # open the image
    image = Image.open(image_path)
    grey_img = image.convert('L')

    image_array = np.asarray(grey_img)
    data = image_array.flatten()
    data = data.reshape(1,-1)
    data_pca = pca.transform(data)
    similar = cosine_similarity(data_pca,X_train_pca)
    max_similarity_index = np.argmax(similar[0])

    image_path = os.path.join(training_set_dir, names_train_set[max_similarity_index])
    res_image = Image.open(image_path)

    grey_res = res_image.convert('L')
    grey_res_val = np.asarray(grey_res)
    res_values = np.asarray(res_image)
    black_img = np.asarray(res_image).copy()
    black_img.fill(0)
    restored_img = restore_colors(image_array, grey_res_val, res_values, 25, black_img)
    plot_results(grey_img, restored_img, res_image, name='sim '+name)


for name in persons:
    # construct the relative path to the image
    image_path = os.path.join(test_set_dir, name + '.' + str(20) + '.jpg')

    # open the image
    image = Image.open(image_path)

    grey_img = image.convert('L')

    avg_image = create_average_image(training_set_dir, name)
    grey_avg_image = np.asarray(Image.fromarray(avg_image.astype('uint8')).convert('L'))
    black_img = np.asarray(avg_image).copy()
    black_img.fill(0)
    restored_img = restore_colors(np.asarray(grey_img), grey_avg_image, avg_image, 25, black_img)
    restored_img = restored_img / 255.0
    # a = Image.fromarray(avg_image.astype('uint8'))
    plot_results(grey_img, restored_img, Image.fromarray(avg_image.astype('uint8')), name ='avg '+ name)



    # # Display the average image
    # plt.imshow(grey_avg_image.astype('uint8'), cmap='gray')
    # plt.show()








