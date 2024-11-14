import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB


def normalize(vector):
    vector -= np.min(vector)
    vector /= np.max(vector)
    return vector


def evaluate(clf, rsk, data, labels):
    scores = cross_val_score(clf, data, labels, cv=rsk, scoring='accuracy')
    return scores.mean(), scores.std()


def zad1(sal_corr):
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    band10 = sal_corr[:, :, 10]
    band100 = sal_corr[:, :, 100]
    band200 = sal_corr[:, :, 200]

    ax[0, 0].imshow(band10, cmap='binary_r')
    ax[0, 0].set_title('Band 10')

    ax[0, 1].imshow(band100, cmap='binary_r')
    ax[0, 1].set_title('Band 100')

    ax[0, 2].imshow(band200, cmap='binary_r')
    ax[0, 2].set_title('Band 200')

    ax[1, 0].plot(sal_corr[10, 10, :])
    ax[1, 1].plot(sal_corr[40, 40, :])
    ax[1, 2].plot(sal_corr[80, 80, :])

    plt.tight_layout()
    plt.savefig("zad1.png")
    plt.show()


def zad2(sal_corr):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    red_c = sal_corr[:, :, 4].astype(float)
    red_c = normalize(red_c)

    green_c = sal_corr[:, :, 12].astype(float)
    green_c = normalize(green_c)

    blue_c = sal_corr[:, :, 26].astype(float)
    blue_c = normalize(blue_c)

    shape = sal_corr.shape
    rgb = np.zeros((shape[0], shape[1], 3))
    rgb[:, :, 0] = red_c
    rgb[:, :, 1] = green_c
    rgb[:, :, 2] = blue_c

    ax[0].imshow(rgb)

    pca_mat = np.reshape(sal_corr, (shape[0] * shape[1], shape[2]))
    pca = PCA(n_components=3)
    components = pca.fit_transform(pca_mat)
    pca_img = np.reshape(components, (shape[0], shape[1], 3))

    red_pca = pca_img[:, :, 0]
    green_pca = pca_img[:, :, 1]
    blue_pca = pca_img[:, :, 2]

    rgb_pca = np.zeros((shape[0], shape[1], 3))
    rgb_pca[:, :, 0] = normalize(red_pca)
    rgb_pca[:, :, 1] = normalize(green_pca)
    rgb_pca[:, :, 2] = normalize(blue_pca)

    ax[1].imshow(rgb_pca)

    plt.tight_layout()
    plt.savefig("zad2.png")
    plt.show()


def zad3(img, labels):
    labels = labels.reshape(-1)

    img_trans = img.reshape(-1, img.shape[2])[labels != 0]
    pca_img = PCA(n_components=3).fit_transform(img.reshape(-1, img.shape[2]))[labels != 0]
    rgb_img = img[:, :, [4, 12, 26]].reshape(-1, 3)[labels != 0]

    clean_labels = labels[labels != 0]

    representations = {'RGB': rgb_img, 'PCA': pca_img, 'IMG': img_trans}

    classifier = GaussianNB()
    rsk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

    for name, rep in representations.items():
        scores = cross_val_score(classifier, rep, clean_labels, cv=rsk, scoring='accuracy')
        print(f'{name} {scores.mean():.3f} ({scores.std():.3f})')


if __name__ == '__main__':
    mat_corr = scipy.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']
    mat_gt = scipy.io.loadmat('SalinasA_gt.mat')['salinasA_gt']
    zad1(mat_corr)
    zad2(mat_corr)
    zad3(mat_corr, mat_gt)
