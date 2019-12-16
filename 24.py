import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from utils import *


def main():
    # Importa dados
    df = pd.read_csv('data.csv', sep=',')

    plot(df['idade'])
    plot(df['peso'])
    plot(df['carga'])
    plot(df['vo2'])


def plot(x):
    distribution = ['expon', 'norm', 'lognorm', 'weibull_min']
    titles = ['Exponencial', 'Gaussiana', 'Lognormal', 'Weibull']
    sparams = [exp_mle(x), norm_mle(x), lognorm_mle(x), weibull_mle(x)]

    f, a = plt.subplots(2, 2)
    a = a.ravel()
    for idx, ax in enumerate(a):
        stats.probplot(x, dist=distribution[idx], sparams=sparams[idx], plot=ax)
        ax.set_title(titles[idx])
    plt.tight_layout()
    plt.show()

    plt.show()


if __name__ == '__main__':
    main()
