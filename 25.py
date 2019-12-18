import pandas as pd
from scipy import stats

from utils import *


def main():
    df = pd.read_csv('data.csv', sep=',')

    print('== Idade ==')
    test(df['idade'])
    print('===========')
    print('== Peso ==')
    test(df['peso'])
    print('===========')
    print('== Carga ==')
    test(df['carga'])
    print('===========')
    print('== VO2 ==')
    test(df['vo2'])
    print('===========')


def test(x):
    expon_params = expon_mle(x)
    ks_expon = stats.kstest(x, 'expon', expon_params)
    print('Exponencial')
    print(f'D = {ks_expon[0]}')
    print(f'p_value = {ks_expon[1]}\n')

    print('Gaussiana')
    norm_params = norm_mle(x)
    ks_norm = stats.kstest(x, 'norm', norm_params)
    print(f'D = {ks_norm[0]}')
    print(f'p_value = {ks_norm[1]}\n')

    print('Lognormal')
    lognorm_params = lognorm_mle(x)
    ks_lognorm = stats.kstest(x, 'lognorm', lognorm_params)
    print(f'D = {ks_lognorm[0]}')
    print(f'p_value = {ks_lognorm[1]}\n')

    print('Weibull')
    weibull_params = weibull_mle(x)
    ks_weibull = stats.kstest(x, 'weibull_min', weibull_params)
    print(f'D = {ks_weibull[0]}')
    print(f'p_value = {ks_weibull[1]}')


if __name__ == '__main__':
    main()
