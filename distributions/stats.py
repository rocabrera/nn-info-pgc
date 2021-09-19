import numpy as np
import pandas as pd


def get_prob_outcomes(x:pd.Series):
    return x.value_counts(normalize=True)

def entropy_metric(prob_outcomes:pd.Series):
    return -prob_outcomes.mul(prob_outcomes.apply(np.log2)).sum()

def get_entropy(x:pd.Series):
    prob_outcomes = get_prob_outcomes(x)
    return entropy_metric(prob_outcomes)