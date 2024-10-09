import numpy as np
import matplotlib.pyplot as plt


def task_1():
    x = np.linspace(0, 4 * np.pi, 40)
    y = np.sin(x)

    y_2d = y[:, np.newaxis] * y[np.newaxis, :]
    min_val = round(np.min(y_2d), 3)
    max_val = round(np.max(y_2d), 3)

    y_2d_norm = y_2d - min_val
    y_2d_norm /= round(np.max(y_2d_norm), 3)
    nmin_val = round(np.min(y_2d_norm), 3)
    nmax_val = round(np.max(y_2d_norm), 3)

    return x, y, (y_2d, min_val, max_val), (y_2d_norm, nmin_val, nmax_val)


def task_2(fun, targets: []):
    response = []
    for target in targets:
        l = np.power(2, target) - 1
        tar_fun = fun * l
        tar_fun = np.rint(tar_fun).astype(int)
        response.append((tar_fun, np.min(tar_fun), np.max(tar_fun)))
    return response


def task_3(fun, targets: []):
    noise = np.random.normal(size=(fun.shape))
    fun_noisy = fun + noise

    results = []
    for number in targets:
        noisy_signals = []
        for i in range(number):
            noise = np.random.normal(size=(fun.shape))
            noisy_signals.append(noise + fun)

        noisy_signals = np.array(noisy_signals)
        avg_signal = np.mean(noisy_signals, axis=0)
        results.append((avg_signal, number))

    return fun_noisy, results


def display():
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    x, y, fold, fold_nom = task_1()

    ax[0, 0].plot(x, y)
    ax[0, 0].set_title('Sin(x)')
    ax[0, 1].imshow(fold[0], cmap='magma')
    ax[0, 1].set_title(f'Min: {fold[1]}, Max: {fold[2]}')
    ax[0, 2].imshow(fold_nom[0], cmap='magma')
    ax[0, 2].set_title(f'Min: {fold_nom[1]}, Max: {fold_nom[2]}')

    response = task_2(fold[0], [2, 4, 8])

    ax[1, 0].imshow(response[0][0], cmap='magma')
    ax[1, 0].set_title(f'Min: {response[0][1]}, Max: {response[0][2]}')
    ax[1, 1].imshow(response[1][0], cmap='magma')
    ax[1, 1].set_title(f'Min: {response[1][1]}, Max: {response[1][2]}')
    ax[1, 2].imshow(response[2][0], cmap='magma')
    ax[1, 2].set_title(f'Min: {response[2][1]}, Max: {response[2][2]}')

    fun_noisy, results = task_3(fold[0], [50, 1000])

    ax[2, 0].imshow(fun_noisy, cmap='magma')
    ax[2, 0].set_title('Noised sin 2d')
    ax[2, 1].imshow(results[0][0], cmap='magma')
    ax[2, 1].set_title(f'n = {results[0][1]}')
    ax[2, 2].imshow(results[1][0], cmap='magma')
    ax[2, 2].set_title(f'n = {results[1][1]}')

    plt.show()


if __name__ == '__main__':
    display()
