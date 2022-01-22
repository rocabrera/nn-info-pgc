import pytest
import pandas as pd
from src.distributions.discrete_stats import (
    entropy_metric,
    get_entropy,
    # get_discrete_mutual_information,
)


@pytest.mark.parametrize(
    "input, expected",
    [
        (pd.Series([0.5, 0.5]), 1),
        (pd.Series([0.5, 0.5, 0]), 1.0),
        (pd.Series([0.5, 0, 0.5]), 1.0),
    ],
)
def test_entropy_metric(input, expected):
    response = entropy_metric(input)
    assert round(response, 6) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (pd.Series([0, 0, 1, 1]), 1),
        (pd.Series([0, 1, 0, 1, 0, 1]), 1.0),
        (pd.Series([2, 7.5, 2, 7.5]), 1.0),
    ],
)
def test_get_entropy(input, expected):
    response = get_entropy(input)
    assert round(response, 6) == expected


def test_get_discrete_mutual_information():
    pass
