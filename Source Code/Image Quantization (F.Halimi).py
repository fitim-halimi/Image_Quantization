print(__doc__)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time


n_colors = 8

# Load the photo
path = r"C:\Users\Fitim Halimi\Desktop\Robot Vision\McM\\"
imgpath =  path + "1.tif"
img = cv2.imread(imgpath)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
img = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(img.shape)
assert d == 3
image_array = np.reshape(img, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Display all results and the original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (256 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image (256 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()


def psnr(img1, img2):
  
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

d=psnr(img,recreate_image(kmeans.cluster_centers_, labels, w, h))
a=psnr(img,recreate_image(codebook_random, labels_random, w, h))

def MSE(img1, img2):
  
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    return mse

c=MSE(img,recreate_image(kmeans.cluster_centers_, labels, w, h))
b=MSE(img,recreate_image(codebook_random, labels_random, w, h))


print("PSNR (original image compared with Kmeans++):")
print(d)
print("MSE (original image compared with Kmeans++):")
print(c)

print("PSNR (original image compared with Random method):")
print(a)
print("MSE (original image compared with Random method):")
print(b)


