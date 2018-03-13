import numpy as np


# ZERO GENERATION
def dropout(X, decay=0, uniform=True):
    """
    X: original testing set
    decay: decay parameter estimated by ZIFA
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = np.copy(X)
    # compute dropout prob
    p = np.exp( - decay * np.log(1 + X_zero)**2)
    # select non-zero subset
    i,j = np.nonzero(X_zero)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    if uniform == False:
        rate = 1-p[i[ix], j[ix]]
    else:
        rate = 0.1
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate )
    return X_zero, i, j, ix


# IMPUTATION METRICS
def imputation_error(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    all_index = i[ix], j[ix]
    x, y = X_mean[all_index], X[all_index]
    return np.median(np.abs(x[X_zero[all_index] == 0] - y[X_zero[all_index] == 0]))
