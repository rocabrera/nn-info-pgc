import numpy as np
import pandas as pd

from general_stats import multivariate_gaussian


def continuos_marginal_entropy(x:pd.Series, kernel, sigma = 1):
    N_x = len(x)
    return (-1/N_x)*sum(np.log2(kernel(i,x, sigma).sum()/N_x) for i in x)


def continuos_join_gaussian_entropy(df:pd.DataFrame, sigma = 1):
    
    N_x = len(df)
    
    H = np.power(sigma, 2) * np.eye(len(df.columns))
    Hinv = np.linalg.pinv(H)
    Hdet = np.linalg.det(H)

    return (-1/N_x) * sum(np.log2(sum(multivariate_gaussian(i, j, Hinv, Hdet) for j in df.values)/N_x) 
                                                                              for i in df.values)


def get_continuos_mutual_info(df, sigma=1):
    
    marginal_entropy = df.apply(continuos_marginal_entropy, 
                                args = [univariate_gaussian, sigma])
    
    join_entropy = continuos_join_gaussian_entropy(df, sigma=sigma)
        
    return sum(marginal_entropy) - join_entropy