import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    df = pd.read_csv('data.csv', sep=',')

    print_corr(df['idade'], df['vo2'])
    reg = linear_regression(df['idade'], df['vo2'])
    print(reg)
    scatter_plot(df['idade'], df['vo2'], reg, 'Scatter Plot - Idade x VO2')

    print('\n')

    print_corr(df['peso'], df['vo2'])
    reg = linear_regression(df['peso'], df['vo2'])
    print(reg)
    scatter_plot(df['peso'], df['vo2'], reg, 'Scatter Plot - Peso x VO2')

    print('\n')

    print_corr(df['carga'], df['vo2'])
    reg = linear_regression(df['carga'], df['vo2'])
    print(reg)
    scatter_plot(df['carga'], df['vo2'], reg, 'Scatter Plot - Carga x VO2')


def print_corr(x, y):
    print(
        f'{x.name} x {y.name} | r = {(np.sum((x - x.mean()) * (y - y.mean()))) / (np.sqrt(np.sum((x - x.mean()) ** 2)) * np.sqrt(np.sum((y - y.mean()) ** 2)))}')


def linear_regression(x, y):
    return np.polyfit(x, y, 1)


def scatter_plot(x, y, reg, title):
    plt.scatter(x, y, alpha=0.5)
    plt.plot(x, x * reg[0] + reg[1], color='red')
    plt.title(title)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.show()


if __name__ == '__main__':
    main()
