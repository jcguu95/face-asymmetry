import math
import scipy
import numpy as np

# Let $$F(x; d_{1}, d_{2}) = I_{d_{1}x/(d_{1}x+d_{2})}(d_{1}/2,
# d_{2}/2),$$ where $I_{x}(a,b)$ denotes the regularized
# incomplete beta function $\frac{B(x;a,b)}{B(a,b)}$ [3][4].
#
# By [1], if $x_1, .., x_n$ are i.i.d. $N(\mu,\sigma^2)$ random
# variables, then $$\Sum_{i=1}^n (x_{i} - \overline{x})^{2} \sim
# sigma^2 \chi_{n-1}^2.$$
#
# By [2], if $X \sim \chi^{2}_{d_{1}}$ and $Y \sim
# \chi^{2}_{d_{2}}$ are independent, then $(X/d_{1})/(Y/d_{2})
# \sim F(d_{1},d_{2})$.
#
# In conclusion, if $x_{1}, .., x_{n} \sim^{iid} N(\mu,\sigma^2)$
# and $y_{1},..,y_{m} \sim^{iid} N(\mu',\sigma^2)$, then the
# test statistics $$T(\vec{x},\vec{y}) = \frac{\Sum_{i=1}^n
# (x_{i} - \overline{x})^2}{\Sum_{i=1}^m (y_{i} -
# \overline{y})^2}$$ follows F(n-1,m-1), characterized by the
# regularized incomplete beta function.

def cdf_F (x,n,m):
    return scipy.special.betainc(n/2,m/2,n*x/(n*x+m))

def sos (ts):
    '''Sum of squares. E.g. [1,2,3] -> 14.'''
    return sum(map((lambda t: t**2), ts))

def sample_average (ts):
    return sum(ts)/len(ts)

def sample_variance_unnormalized (ts):
    '''Sample variance estimator, without dividing by (len(ts) -
    1). If ts is sampled from $N(\mu,\sigma^2)$, then the result
    should follow $\chi_{n-1}^2$.'''
    assert(len(ts) > 1)
    E = sample_average(ts)
    return sos(map((lambda t: t-E), ts))

def sample_variance (ts):
    return sample_variance_unnormalized(ts) / len(ts)

# Assume xs and ys are sampled respectively from
# N(mu,sigma^2) and N(mu',sigma'^2). We want to use F-test to
# construct a hypothesis test and argue whether sigma^2 ==
# sigma'^2. For details see
# https://en.wikipedia.org/wiki/F-distribution
#
# We set the null and alternative hypotheses to be
#
#   H0: \sigma^2 == \sigma'^2
#   H1: \sigma^2 != \sigma'^2
#
# If H0 is true, then log(ratio) must be zero. We construct
# the p-value by measure how far it deviates from zero:
#
#   P := P(|log(ratio)| >= t)
#      = P(ratio >= e^t) + P(ratio <= e^(-t))
#
# If H0 is true, then the ratio has the chi-squared
# distribution on k := len(xs) freedoms. By wikipedia, the
# cumulative distribution function (CDF) (in x) of that is
# given by
#
#   F(x; k) = lower-incomplete-gamma(k/2, x/2)
#
# Finally, we define the p-value. If it is lower than 0.05,
# we should reject H0 and accept H1.
def f_test_1D (xs, ys):
    '''Run F-test for the input arrays.'''
    n     = len(xs)
    m     = len(ys)
    ratio = (sample_variance_unnormalized(xs)/n) / \
            (sample_variance_unnormalized(ys)/m)
    print("f_test_1D: ratio: ", ratio)
    t     = np.absolute(math.log(ratio))
    p_value =      cdf_F(n-1,m-1, math.exp(-t)) + \
              (1 - cdf_F(n-1,m-1, math.exp( t)))
    return p_value

### Testing..
def test_f_test_1D (cov1,cov2):
    mean = 0
    Xs = np.random.normal(mean,cov1,1000)
    Ys = np.random.normal(mean,cov2,1000)
    return f_test_1D(Xs,Ys)
# I'm worried.. the ratio cov1/cov2 (assuming > 1) needs to be
# larger than ~10 for this test to reject H0.

##############################################################

# We have two sets of n vectors in $R^p$ sampled from multivariate
# normal distributions N(\mu,\Sigma) and N(\mu',\Sigma'). We want
# to compare their largest eigenvalues \lambda_1, \lambda_1'
# respectively by the following method: Compute their first
# princple component v, orthogonally project the vectors onto v,
# treat the lengths to the sample average as the data generated
# by N(0,\sigma^2=\lambda_1(')). Then, we feed the two sets of
# lengths into F-test. If the p-value is less than 0.05, we
# reject H0, accept H1, and conclude that \lambda_1 is not
# \lambda_1'.
def one_dimensionalize (Xs):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(Xs)
    mean = pca.mean_
    cpnt = pca.components_[0]
    result = []
    for X in Xs:
        u = (X-mean)
        v = cpnt
        result.append(np.dot(u,v)/(np.dot(v,v)**(1/2)))
    return result

def f_test_higherD_naive (Xs, Ys):
    '''Main function: We project the X's onto the first principal
    component of Xs (similarly for Y's), and run the 1D f-test on
    the lengths of the projected vectors.'''
    n = len(Xs)
    p = len(Xs[0])
    assert(len(Xs)==len(Ys))
    assert(len(Xs[0])==len(Ys[0]))
    for k in range(n):
        assert(len(Xs[0])==len(Xs[k]))
        assert(len(Ys[0])==len(Ys[k]))
    Xs1D = one_dimensionalize(Xs)
    Ys1D = one_dimensionalize(Ys)
    p_value = f_test_1D(Xs1D,Ys1D)
    return p_value

# My observation is that |log(var1/var2)| has to be larger than
# log(10) for the p_value to be < 0.05. Maybe it's not so bad.
def test_ (var1, var2):
    '''Run the test for N(0,I) in 2000 dimensions.'''
    mean = 0
    Xs = ([[np.random.normal(mean,var1) for p in range(2000)] for n in range(20)])
    Ys = ([[np.random.normal(mean,var2) for p in range(2000)] for n in range(20)])
    return f_test_higherD_naive(Xs,Ys)

# My observation is that |log(var1/var2)| has to be larger than
# log(100) for the p_value to be < 0.05. It means that our method
# here is too naive bad.
def test_2 (var1, var2):
    '''Test it in 2D.'''
    mean = [0, 0]
    cov1 = [[var1, 0], [0, 1]]
    cov2 = [[var2, 0], [0, 1]]
    Xs = np.random.multivariate_normal(mean,cov1,100)
    Ys = np.random.multivariate_normal(mean,cov2,100)
    return f_test_higherD_naive(Xs,Ys)

# # To test it in higher dimensional with genuine covariance
# # matrix, I wrote the following function. However, it is way too
# # slow, even for dimension 20.
# def higher_normal (mean, cov):
#     '''Generate a random vector from N(mean,cov) using Monte Carlo method. It's slow.'''
#     # Health check
#     k = len(mean) # dimension
#     assert(k == len(cov))
#     for row in cov:
#         assert(k == len(row))
#     # assert(is_positive_definite(cov)) # TODO add this back later
#     # Calculation
#     mean = np.array(mean)
#     cov = np.array(cov)
#     def pdf (x):
#     # Formulae taken from
#     # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
#         x = np.array(x)
#         denom = (2*np.pi)**(k/2) * np.linalg.det(cov)**(1/2)
#         numer = np.exp((-1/2) * (x-mean) @ np.linalg.inv(cov) @ (x-mean).T)
#         return numer/denom
#     max_std = max(np.linalg.eigvals(cov))**(1/2)
#     import random
#     while True:
#         value  = random.uniform(0,1)
#         vector = [random.uniform(-10*max_std,10*max_std) for i in range(k)]
#         if pdf(vector) >= value:
#             return vector
#         else:
#             continue

# # testing higher_normal (only for 1D)
# from scipy import stats
# def is_normal_1D (xs):
#     "This is a statistical test for the normality of xs."
#     return stats.normaltest(xs)[1] > 0.05

# # If the following is smaller than 0.05, then we can conclude
# # higher_normal doesn't always give normal distributed data.
# assert(is_normal_1D([higher_normal([0],[[1]])[0] for i in range(1000)]))

# is_normal_1D(one_dimensionalize([higher_normal([0,0],[[1,0],[0,1]]) for i in range(1000)]))
# # is_normal_1D(one_dimensionalize([higher_normal(np.zeros(200),np.identity(200)) for i in range(30)])) ## this takes too long

# is_normal_1D(one_dimensionalize([[np.random.normal(0,1) for p in range(2000)] for n in range(20)]))

# def is_positive_definite (M):
#     '''Check if the matrix M is positive definite using the
#     Principal Minor Test. Code is taken from the link below - I
#     haven't checked its credibility.
#     https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy

#     '''
#     import numpy as np
#     row = len(M)
#     i = 0
#     j = 0
#     for i in range(row+1):
#         Step = M[:i,:j]
#         j += 1
#         i += 1
#         det = np.linalg.det(Step)
#         if det > 0:
#             continue
#         else:
#             return False
#     return True

# References
#
# + [1] https://en.wikipedia.org/wiki/Chi-squared_distribution
#
# + [2] https://en.wikipedia.org/wiki/F-distribution
#
# + [3] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
#
# + [4] https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainc.html
