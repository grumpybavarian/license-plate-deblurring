from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

img = imread("./Data/Original_Train_Data/img_0008.jpg")
# img_res = imresize(img, (int(img.shape[0] * 0.25), int(img.shape[1] * 0.25), 3))
# img_res = imresize(img, img.shape)
pca = PCA(n_components=60)
img_pca = pca.fit_transform(img[:,:, 0])
img_pca = pca.inverse_transform(img_pca)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_pca)
plt.show()
