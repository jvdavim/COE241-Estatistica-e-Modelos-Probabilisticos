import math

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


def main():
    # Importa dados
    df = pd.read_csv('data.csv', sep=',')
    # Desenha histogramas
    get_histogram(df)
    # Desenha funcao distribuicao empirica
    get_ecdf(df)


def get_histogram(df):
    # Calcula numero de amostras
    n = len(df)
    n_idade = n
    n_peso = n
    n_carga = n
    n_vo2 = n

    # Desenha histogramas
    titles = ['Idade', 'Peso', 'Carga', 'VO2']
    bins = [get_bin(n_idade), get_bin(n_peso), get_bin(n_carga), get_bin(n_vo2)]

    f, a = plt.subplots(2, 2)
    a = a.ravel()
    for idx, ax in enumerate(a):
        ax.hist(df.iloc[:, idx], bins[idx])
        ax.set_title(titles[idx])
    plt.tight_layout()
    plt.show()


def get_bin(n):
    return round(1 + 3.3 * math.log10(n))


def get_ecdf(df):
    titles = ['Idade', 'Peso', 'Carga', 'VO2']
    ecdf = [ECDF(df['idade']), ECDF(df['peso']), ECDF(df['carga']), ECDF(df['vo2'])]

    f, a = plt.subplots(2, 2)
    a = a.ravel()
    for idx, ax in enumerate(a):
        ax.plot(ecdf[idx].x, ecdf[idx].y)
        ax.set_title(titles[idx])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
