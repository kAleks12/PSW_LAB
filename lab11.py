import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import disk
from skimage.measure import label
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def main():
    # Task 1
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    img = np.zeros((100, 100, 3))
    gt = np.zeros((100, 100))
    for _ in range(3):
        center = (np.random.randint(40, 60), np.random.randint(40, 60))
        rad = np.random.randint(10, 40)
        rr, cc = disk(center, rad, shape=img.shape)
        choice = np.random.choice([0, 1, 2])
        change = np.random.randint(100, 255)
        img[rr, cc, choice] += change
        gt[rr, cc] += change

    gt = label(gt)
    noise = np.random.normal(0, 16, img.shape)
    img = np.clip(img + noise, 0, 255).astype(int)

    ax[0].imshow(img)
    ax[1].imshow(gt)

    plt.show()

    # Task 2

    img_res = img.reshape(-1, img.shape[2])
    x, y = np.meshgrid(range(img.shape[0]), range(img.shape[1]), indexing='ij')
    x = x.flatten()
    y = y.flatten()

    X = np.column_stack((img_res, x, y)).astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = gt.flatten()

    print(X.shape, y.shape)
    print(X[1, :], y[1])

    # Task 3

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    models = [KMeans(), MiniBatchKMeans(), Birch(), DBSCAN()]
    results = []
    for model in models:
        result = model.fit_predict(X)
        results.append(result)

    row = 0
    column = 2

    ax[0, 0].imshow(img)
    ax[0, 1].imshow(gt)

    for pred in results:
        part_img = pred.reshape(gt.shape)
        score = round(adjusted_rand_score(y, pred), 4)
        ax[row, column].imshow(part_img)
        ax[row, column].set_title(f"{score}")

        if column == 2:
            column = 0
            row += 1
        else:
            column += 1

    plt.show()


if __name__ == "__main__":
    main()
