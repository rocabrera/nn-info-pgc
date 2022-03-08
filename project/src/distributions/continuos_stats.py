import numpy as np
import pandas as pd
from distributions.general_stats import multivariate_gaussian

# import pdb
# def continuos_marginal_entropy(x: pd.Series, kernel, sigma=1):
#     N_x = len(x)
#     return (-1 / N_x) * sum(np.log2(kernel(i, x, sigma).sum() / N_x) for i in x)

def continuos_join_gaussian_entropy(array: np.array, kernel_size=1):

    """
    N_x define é o número de amostras em um batch
    n é o tamanho do vetor X, T ou XT

    Exemplo suponha uma rede com: 
    - arquitetura: [10, 5]
    - número de amostras: 1000 (batch)
    - número de features: 2

    Temos 3 camadas 
    T: [(1000, 10), (1000,5), (1000, 1)]
    I(Y, T): [(1000, 11), (1000,6), (1000, 2)]

    O número de colunas muda por isso eu não consigo tirar de dentro do for de forma simples
    a não ser que eu calcula-se todos Hinv e Hdet coloca-se em alguma estrutura de dados e passa-se isso
    na hora de calcular.

    
    1- Vetorizar a função multivariate_gaussian!
    2- Jogar para fora o Hinv e Hdet
    3- NORMALIZAR 
    """
    nrows, ncols = array.shape

    H = np.power(kernel_size, 2) * np.eye(ncols)
    Hinv = np.linalg.pinv(H)
    Hdet = np.power(np.power(kernel_size, 2), ncols)  # nesse caso em especifico
    # pdb.set_trace()
    
    return (-1 / nrows) * np.sum(
        np.log2(np.sum(multivariate_gaussian(i, j, Hinv, Hdet) for j in array) / nrows)
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
