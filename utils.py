import math

from scipy.stats import weibull_min


def exp_mle(x):
    return len(x) / x.sum()


def norm_mle(x):
    mean = x.mean()
    var = 0
    for i, v in x.iteritems():
        var += (v - mean) ** 2
    var = var / len(x)
    return mean, var


def lognorm_mle(x):
    mean = 0
    for i, v in x.iteritems():
        mean += math.log(v)
    mean = mean / len(x)
    var = 0
    for i, v in x.iteritems():
        var += (math.log(v) - mean) ** 2
    var = var / len(x)
    return mean, var


def weibull_mle(x):
    # Shape, location, scale
    return weibull_min.fit(x, floc=0)
