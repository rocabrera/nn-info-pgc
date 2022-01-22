import numpy as np


def univariate_gaussian(x: int, mu: float, sig: float):
    numerator = np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2)))
    denominator = sig * np.sqrt(2 * np.pi)
    return numerator / denominator


def multivariate_gaussian(x, y, Hinv, Hdet):
    expoente = (-1 / 2) * np.dot(np.dot((x - y).T, Hinv), (x - y))
    denominador = np.power(2 * np.pi, len(x)) * Hdet
    return np.exp(expoente) / np.sqrt(denominador)
