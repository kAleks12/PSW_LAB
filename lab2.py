import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
import skimage as ski


def execute():
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    chelsea = ski.data.chelsea()
    ax[0, 0].imshow(chelsea)

    chelsea_mean = np.mean(chelsea, axis=2)
    chelsea_mr = chelsea_mean[::8, ::8]
    ax[0, 1].imshow(chelsea_mr, cmap='binary_r')

    rotation_15 = [[np.cos(np.pi / 12), -np.sin(np.pi / 12), 0], [np.sin(np.pi / 12), np.cos(np.pi / 12), 0], [0, 0, 1]]
    tform = ski.transform.AffineTransform(matrix=rotation_15)
    warped = ski.transform.warp(chelsea_mr, inverse_map=tform.inverse)
    ax[1, 0].imshow(warped, cmap='binary_r')

    x_rotation = [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]
    tform = ski.transform.AffineTransform(matrix=x_rotation)
    warped = ski.transform.warp(chelsea_mr, inverse_map=tform.inverse)
    ax[1, 1].imshow(warped, cmap='binary_r')
    plt.show()

    # zadanie 2
    fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)
    x, y = chelsea_mr.shape
    xx, yy = np.meshgrid(range(0, x), range(0, y))
    xx = xx.T.flatten()
    yy = yy.T.flatten()
    points = np.array([xx, yy])
    points = points.T
    ax[0].scatter(points[:, 0], points[:, 1], c=chelsea_mr.flatten(), cmap='binary_r')

    points_ext = np.column_stack((points, np.ones(2166)))
    new_points = points_ext @ rotation_15
    ax[1].scatter(new_points[:, 0], new_points[:, 1], c=chelsea_mr.flatten(), cmap='binary_r')

    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(10, 7), sharex=True, sharey=True)
    ax[0].scatter(points[:, 0], points[:, 1], c=chelsea_mr.flatten(), cmap='binary_r')

    random_indices = np.random.choice(points.shape[0], 1000)
    random_points = points[random_indices, :]
    random_img = chelsea_mr.flatten()[random_indices]
    ax[1].scatter(random_points[:, 0], random_points[:, 1], c=random_img, cmap='binary_r')

    xx, yy = np.meshgrid(np.linspace(0, 38, 300), np.linspace(0, 57, 400))
    xx = xx.T.flatten()
    yy = yy.T.flatten()
    inter_points = np.array([xx, yy]).T

    distances = scipy.spatial.distance.cdist(inter_points, random_points)
    closest_dx = np.argmin(distances, 1)
    closest_intensities = random_img[closest_dx]

    ax[2].scatter(inter_points[:, 0], inter_points[:, 1], c=closest_intensities, cmap='binary_r')

    plt.show()


if __name__ == '__main__':
    execute()
