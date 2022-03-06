import numpy as np
import pandas as pd
from distributions.general_stats import multivariate_gaussian


# def continuos_marginal_entropy(x: pd.Series, kernel, sigma=1):
#     N_x = len(x)
#     return (-1 / N_x) * sum(np.log2(kernel(i, x, sigma).sum() / N_x) for i in x)


def continuos_join_gaussian_entropy(df: pd.DataFrame, kernel_size=1):

    N_x = len(df)
    H = np.power(kernel_size, 2) * np.eye(len(df.columns))
    Hinv = np.linalg.pinv(H)
    Hdet = np.linalg.det(H)
    return (-1 / N_x) * sum(
        np.log2(sum(multivariate_gaussian(i, j, Hinv, Hdet) for j in df.values) / N_x)
        for i in df.values
    )


def get_continuos_mutual_information(df: pd.DataFrame, kernel_size: int, n_fst_vector: int):
    marginal_entropy_x = continuos_join_gaussian_entropy(
        df = df.iloc[:, :n_fst_vector],
        kernel_size = kernel_size,
    )
    marginal_entropy_y = continuos_join_gaussian_entropy(
        df = df.iloc[:, n_fst_vector:],
        kernel_size = kernel_size,
    )

    join_entropy = continuos_join_gaussian_entropy(df, kernel_size=kernel_size)

    return marginal_entropy_x + marginal_entropy_y - join_entropy
