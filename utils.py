import scipy.stats


def expon_mle(x):
    return scipy.stats.expon.fit(x, floc=0)


def norm_mle(x):
    return scipy.stats.norm.fit(x)


def lognorm_mle(x):
    return scipy.stats.lognorm.fit(x, floc=0)


def weibull_mle(x):
    return scipy.stats.weibull_min.fit(x, floc=0)
