import numpy as np
import pandas as pd


def entropy_metric(prob_outcomes: pd.Series):
    """Calculate the following formula H(X) = - sum(p(x) * log2 p(x)).

    Args:
      prob_outcomes: Series with the probability of every possible outcome.

    Returns:
        H(X) = - sum(p(x) * log p(x))
    """
    return -prob_outcomes.mul(prob_outcomes.apply(np.log2)).sum()


def get_entropy(x: pd.Series):
    """Returns Entropy of a Random Variable.

    Args:
      x: Series with the samples.

    Returns:
      Entropy - H(x)
    """
    prob_outcomes = x.value_counts(normalize=True)
    return entropy_metric(prob_outcomes)


def get_discrete_mutual_information(df: pd.DataFrame, n_bin: int, n_fst_vector: int):

    bins_df = df.apply(lambda x: pd.cut(x, n_bin, labels=False))
    sample_size = len(bins_df)
    marginal_entropy_x = entropy_metric(
        bins_df.iloc[:, :n_fst_vector]
        .groupby(bins_df.columns.tolist()[:n_fst_vector])
        .size()
        .div(sample_size)
    )

    marginal_entropy_y = entropy_metric(
        bins_df.iloc[:, n_fst_vector:]
        .groupby(bins_df.columns.tolist()[n_fst_vector:])
        .size()
        .div(sample_size)
    )

    joint_entropy_xy = entropy_metric(
        bins_df.groupby(bins_df.columns.tolist()).size().div(sample_size)
    )

    return marginal_entropy_x + marginal_entropy_y - joint_entropy_xy
