import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF

from utils import *


def main():
    # Importa dados
    df = pd.read_csv('data.csv', sep=',')

    plot(df['idade'])
    plot(df['peso'])
    plot(df['carga'])
    plot(df['vo2'])


def plot(data):
    print(f'\n== {data.name} ==')
    expon_loc, expon_scale = expon_mle(data)
    print(f'-- Exponencial --')
    print(f'Lambda: {1 / expon_scale}')
    print(f'-----------------')
    norm_loc, norm_scale = norm_mle(data)
    print(f'-- Gaussiana --')
    print(f'Média: {norm_loc}')
    print(f'Variância: {np.power(norm_scale, 2)}')
    print(f'-----------------')
    lognorm_shape, lognorm_loc, lognorm_scale = lognorm_mle(data)
    print(f'-- Lognormal --')
    print(f'Média: {np.sum(np.log(data)) / data.count()}')
    print(f'Variância: {np.power(lognorm_shape, 2)}')
    print(f'-----------------')
    weibull_shape, weibull_loc, weibull_scale = weibull_mle(data)
    print(f'-- Weibull --')
    print(f'Shape: {weibull_shape}')
    print(f'Location: {weibull_loc}')
    print(f'Scale: {weibull_scale}')
    print(f'-----------------')
    print(f'=================')

    ecdf = ECDF(data)
    plt.plot(ecdf.x, ecdf.y, label='Distribuição Empírica')

    x = np.linspace(data.min(), data.max(), 1000)

    y = scipy.stats.expon.cdf(x, loc=expon_loc, scale=expon_scale)
    plt.plot(x, y, label='Exponencial')

    y = scipy.stats.norm.cdf(x, loc=norm_loc, scale=norm_scale)
    plt.plot(x, y, label='Gaussiana')

    y = scipy.stats.lognorm.cdf(x, lognorm_shape, loc=lognorm_loc, scale=lognorm_scale)
    plt.plot(x, y, label='Lognormal')

    y = scipy.stats.weibull_min.cdf(x, weibull_shape, loc=weibull_loc, scale=weibull_scale)
    plt.plot(x, y, label='Weibull')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
