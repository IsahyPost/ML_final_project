from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from matplotlib import pyplot as plt
import math
import os


persons = ['cgboyc', 'cmkirk', 'djhugh','dmwest', 'gmwate','khughe','lejnno']

for name in persons:
    imageName = name + '.' + str(20) + '.jpg'
    # Open the image form working directorys
    image = Image.open(imageName).convert('L')
    plt.imshow(image, cmap='gray')
    plt.show()


# Load the images into a list
image_folder = "path/to/folder/containing/images"
image_files = os.listdir(image_folder)
images = []
for image_file in image_files:
    if image_file.endswith(".jpg"):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('L')
        images.append(np.array(image))

# Reshape the images into 1D arrays
X = np.array(images).reshape(len(images), -1)

# Perform PCA
pca = PCA(n_components=X.shape[0])
X_pca = pca.fit_transform(X)

