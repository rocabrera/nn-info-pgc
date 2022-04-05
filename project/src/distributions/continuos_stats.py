import numpy as np
import pandas as pd
from distributions.general_stats import multivariate_gaussian

# import pdb
# def continuos_marginal_entropy(x: pd.Series, kernel, sigma=1):
#     N_x = len(x)
#     return (-1 / N_x) * sum(np.log2(kernel(i, x, sigma).sum() / N_x) for i in x)

def continuos_join_gaussian_entropy(array: np.array, kernel_size=1):

    nrows, ncols = array.shape

    H = np.power(kernel_size, 2) * np.eye(ncols)
    Hinv = (1/np.power(kernel_size,2))*np.eye(ncols).astype(np.float32) # nesse caso em especifico
    Hdet = np.power(np.power(kernel_size, 2), ncols)  # nesse caso em especifico
    # pdb.set_trace()

    return (-1 / nrows) * sum(
        np.log2(sum(multivariate_gaussian(i, j, Hinv, Hdet) for j in array) / nrows)
        for i in array
    )


def get_continuos_mutual_information(array: np.array, kernel_size: int, n_fst_vector: int):
    marginal_entropy_x = continuos_join_gaussian_entropy(
        array = array[:, :n_fst_vector],
        kernel_size = kernel_size,
    )
    marginal_entropy_y = continuos_join_gaussian_entropy(
        array = array[:, n_fst_vector:],
        kernel_size = kernel_size,
    )

    join_entropy = continuos_join_gaussian_entropy(array, kernel_size=kernel_size)

    return marginal_entropy_x + marginal_entropy_y - join_entropy
