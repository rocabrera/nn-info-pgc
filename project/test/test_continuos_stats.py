import pytest
import pandas as pd
from src.distributions.general_stats import multivariate_gaussian
from src.distributions.continuos_stats import continuos_join_gaussian_entropy



@pytest.mark.parametrize(
    "input, expected",
    [
        ((pd.DataFrame([0, 0, 1, 1]), 2), 1),
    ],
)
def test_continuos_join_gaussian_entropy(input, expected):
    df, kernel_size = input
    response = continuos_join_gaussian_entropy(df=df, 
                                               kernel_size=kernel_size)


def test_multivariate_gaussian():
    pass
