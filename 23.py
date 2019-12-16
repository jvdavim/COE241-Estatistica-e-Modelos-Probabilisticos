import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF

from utils import *


def main():
    # Importa dados
    df = pd.read_csv('data.csv', sep=',')

    # Idade
    exp_lambda_idade = exp_mle(df['idade'])
    norm_mean_idade, norm_var_idade = norm_mle(df['idade'])
    lognorm_mean_idade, lognorm_var_idade = lognorm_mle(df['idade'])
    shape_idade, location_idade, scale_idade = weibull_mle(df['idade'])
    teste = weibull_mle(df['idade'])

    ecdf = ECDF(df['idade'])
    plt.plot(ecdf.x, ecdf.y, label='Distribuição Empírica')

    x = np.linspace(df['idade'].min(), df['idade'].max(), 1000)

    y = scipy.stats.expon.cdf(x, scale=1 / exp_lambda_idade)
    plt.plot(x, y, label='Exponencial')

    y = scipy.stats.norm.cdf(x, loc=norm_mean_idade, scale=math.sqrt(norm_var_idade))
    plt.plot(x, y, label='Gaussiana')

    y = scipy.stats.lognorm.cdf(x, s=math.sqrt(lognorm_var_idade), scale=math.exp(lognorm_mean_idade))
    plt.plot(x, y, label='Lognormal')

    y = scipy.stats.weibull_min.cdf(x, shape_idade, loc=location_idade, scale=scale_idade)
    plt.plot(x, y, label='Weibull')

    plt.legend()
    plt.show()

    # Peso
    exp_lambda_peso = exp_mle(df['peso'])
    norm_mean_peso, norm_var_peso = norm_mle(df['peso'])
    lognorm_mean_peso, lognorm_var_peso = lognorm_mle(df['peso'])
    shape_peso, location_peso, scale_peso = weibull_mle(df['peso'])

    ecdf = ECDF(df['peso'])
    plt.plot(ecdf.x, ecdf.y, label='Distribuição Empírica')

    x = np.linspace(df['peso'].min(), df['peso'].max(), 1000)

    y = scipy.stats.expon.cdf(x, scale=1 / exp_lambda_peso)
    plt.plot(x, y, label='Exponencial')

    y = scipy.stats.norm.cdf(x, loc=norm_mean_peso, scale=math.sqrt(norm_var_peso))
    plt.plot(x, y, label='Gaussiana')

    y = scipy.stats.lognorm.cdf(x, s=math.sqrt(lognorm_var_peso), scale=math.exp(lognorm_mean_peso))
    plt.plot(x, y, label='Lognormal')

    y = scipy.stats.weibull_min.cdf(x, shape_peso, loc=location_peso, scale=scale_peso)
    plt.plot(x, y, label='Weibull')

    plt.legend()
    plt.show()

    # Carga
    exp_lambda_carga = exp_mle(df['carga'])
    norm_mean_carga, norm_var_carga = norm_mle(df['carga'])
    lognorm_mean_carga, lognorm_var_carga = lognorm_mle(df['carga'])
    shape_carga, location_carga, scale_carga = weibull_mle(df['carga'])

    ecdf = ECDF(df['carga'])
    plt.plot(ecdf.x, ecdf.y, label='Distribuição Empírica')

    x = np.linspace(df['carga'].min(), df['carga'].max(), 1000)

    y = scipy.stats.expon.cdf(x, scale=1 / exp_lambda_carga)
    plt.plot(x, y, label='Exponencial')

    y = scipy.stats.norm.cdf(x, loc=norm_mean_carga, scale=math.sqrt(norm_var_carga))
    plt.plot(x, y, label='Gaussiana')

    y = scipy.stats.lognorm.cdf(x, s=math.sqrt(lognorm_var_carga), scale=math.exp(lognorm_mean_carga))
    plt.plot(x, y, label='Lognormal')

    y = scipy.stats.weibull_min.cdf(x, shape_carga, loc=location_carga, scale=scale_carga)
    plt.plot(x, y, label='Weibull')

    plt.legend()
    plt.show()

    # VO2
    exp_lambda_vo2 = exp_mle(df['vo2'])
    norm_mean_vo2, norm_var_vo2 = norm_mle(df['vo2'])
    lognorm_mean_vo2, lognorm_var_vo2 = lognorm_mle(df['vo2'])
    shape_vo2, location_vo2, scale_vo2 = weibull_mle(df['vo2'])

    ecdf = ECDF(df['vo2'])
    plt.plot(ecdf.x, ecdf.y, label='Distribuição Empírica')

    x = np.linspace(df['vo2'].min(), df['vo2'].max(), 1000)

    y = scipy.stats.expon.cdf(x, scale=1 / exp_lambda_vo2)
    plt.plot(x, y, label='Exponencial')

    y = scipy.stats.norm.cdf(x, loc=norm_mean_vo2, scale=math.sqrt(norm_var_vo2))
    plt.plot(x, y, label='Gaussiana')

    y = scipy.stats.lognorm.cdf(x, s=math.sqrt(lognorm_var_vo2), scale=math.exp(lognorm_mean_vo2))
    plt.plot(x, y, label='Lognormal')

    y = scipy.stats.weibull_min.cdf(x, shape_vo2, loc=location_vo2, scale=scale_vo2)
    plt.plot(x, y, label='Weibull')

    plt.legend()
    plt.show()

    print("debug")


if __name__ == '__main__':
    main()
