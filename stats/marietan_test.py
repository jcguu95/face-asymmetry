import math
import scipy
import numpy as np

# We implement the test statistics T_{1} in [5] in order to tell
# if two covariance matrices are different. Recall that the paper
# assumes a condition for our Wishart matrices to satisfy (page
# 4, before sec. 2.2). Assume that our spike model is of order
# $k$ (default to 3), that $n \geq m$, and that $X \in
# \mathbb{R}^{p \times n}$ and $Y \in \mathbb{R}^{p \times m}$
# are centralized data. Then we define
#
#  $$\hat{\Sigma_{X}} = \frac{1}{m}XX^t$$
#  $$\hat{\Sigma_{Y}} = \frac{1}{n}YY^t$$
#
# For each $\hat{\Sigma}$, order its eigenvalues from the largest
# to the smallest $\hat{\lambda}_{i}$. For $1\leq i \leq k$, we
# also denote it by $\hat{\theta}_{i}$ [5, page 10].
#

# k means the number of eigenvalues that much larger than 1 in the true covarivance matrix
# Some Test cases

"""
Example Usage:
x = np.random.rand(100, 22)
x[:, 0:3] *= 100
y = np.random.rand(100, 22)

x = np.random.rand(100, 1000)
x[:, 0:3] *= 100
y = np.random.rand(100, 1000)

"""

_k = 3 # This is the order in the spike model.

# TODO May want to cache this function.
def ordered_eigens (M):
    '''We order the eigenvalues from the largest to smallest.'''
    eigvals, eigvecs = np.linalg.eig(M)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    return {"eigvals": eigvals, "eigvecs": eigvecs}

def hat_Sigma(X):
    m = len(X)
    X = np.array(X)
    return X @ X.T / m

# Then we define $\hat{\hat{\theta_s}} (s = 1 \ldots k)$ do be
# the following.
def hat_hat_theta (s, hat_Sigma):
    '''The unbiased estimators of \theta_s; see [5, Def 2.2].'''
    assert(0<=s<=_k)
    m = len(hat_Sigma)
    eigvals = ordered_eigens(hat_Sigma)["eigvals"]
    denom  = 0
    for i in range(_k,m):
        denom += eigvals[i] / (eigvals[s] - eigvals[i])
    return 1 + (m-_k)/denom

def hat_hat_Sigma (hat_Sigma):
    '''The filtered estimated covariance matrix; see [5, Def 2.2].'''
    eigvecs = ordered_eigens(hat_Sigma)["eigvecs"]
    m = len(hat_Sigma)
    result = np.identity(m)
    for i in range(_k):
        result = result + hat_hat_theta(i, hat_Sigma) * eigvecs[i].T @ eigvecs[i]
    return result

def M (s1, s2, X, rho):
    '''[5, sec 2.5]'''
    m = len(X)
    eigvals = ordered_eigens(hat_Sigma(X))["eigvals"]
    result = 0
    for i in range(_k,m):
        result += eigvals[i]**s1 / (rho - eigvals[i])**s2
    return result / (m-_k)

def sigma_square (X, rho):
    '''[5. section 2.4, p.6]'''
    return 2 * (M(2,2,X,rho) - M(1,1,X,rho)**2) / M(1,1,X,rho)**4

def T_1 (X, Y):
    '''The test statistics $T_1$ in [5, page 5], where the
    denominator $\sigma$ is given on page 6, and where $\rho$ is
    replaced with $\hat{\theta}$ (which is just $\hat{\theta}$ by
    page 10), and where M is given under sec 2.5.

    The main test given in [5] states that under the null
    hypothesis H_0 therein, this is going to be ~ $\chi_k^2$, the
    chi-square distribution of k freedoms.
    '''
    m          = len(X)
    result     = 0
    eigvals_X  = ordered_eigens(hat_Sigma(X))["eigvals"]
    eigvals_Y  = ordered_eigens(hat_Sigma(Y))["eigvals"]
    # TODO can be more efficient by avoiding recomputing the eigens of X and Y.
    for i in range(_k):
        numer   = (hat_hat_theta(i, hat_Sigma(X)) - hat_hat_theta(i, hat_Sigma(Y)))**2
        denom   = sigma_square(X, eigvals_X[i])   + sigma_square(Y, eigvals_Y[i]) # p.6
        result += numer / denom
    return m * result

def centralize (X):
    X = np.array(X)
    mean = 0
    for x in X:
        mean = mean + x
    mean = mean / len(X)
    return X - np.array([mean for i in range(len(X))])

def marietan_T1_test (X, Y):
    '''Based on [5, sec 2.4], if H0 is satisfied, then $T1 \sim
    \chi^2_k$, whose cdf is the incomplete gamma function [6][7],
    and T1 should be as close as to zero. We therefore construct
    the p-value by measuring how far it is from zero.'''
    X       = centralize(X)
    Y       = centralize(Y)
    a       = _k/2
    b       = np.real(T_1(X,Y)/2)
    p_value = 1 - scipy.special.gammainc(a,b)
    return p_value

# References
#
# + [5] https://arxiv.org/pdf/2002.12703.pdf - This is "II". Some
# more details may be found in part "I"
# (https://arxiv.org/pdf/2002.12741.pdf) and an author's thesis.
#
# + [6] https://en.wikipedia.org/wiki/Chi-squared_distribution#Cumulative_distribution_function
#
# + [7] https://en.wikipedia.org/wiki/Incomplete_gamma_function
