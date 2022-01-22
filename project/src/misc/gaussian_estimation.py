import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


X = [2, 5, 9, 10]
sigmas = [1.5, 2, 2.5, 3]
x_values = np.linspace(-7, 20, 120)

fig, axs = plt.subplots(2, 2, figsize=(15, 8))

for ax, sigma in zip(axs.flatten(), sigmas):
    ys = np.zeros(x_values.shape)
    for val in X:
        y = gaussian(x_values, val, sigma)
        ys += y
        ax.plot(x_values, y, c="k", lw=2)

        ax.stem(X, [0.1 for _ in X], "k", markerfmt=" ")

        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        # ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_title(f"Tamanho do kernel: {sigma}", fontsize=16)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Função densidade de probabilidade", fontsize=12)

    ax.plot(x_values, ys, c="r", lw=2)

plt.tight_layout()
plt.show()
