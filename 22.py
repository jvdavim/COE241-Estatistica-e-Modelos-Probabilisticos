import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Importa dados
    df = pd.read_csv('data.csv', sep=',')

    calc_mean_and_var(df)

    boxplot(df)


def calc_mean_and_var(df):
    # Calcula medias e variancias
    mean_idade = df['idade'].mean()
    var_idade = df['idade'].var()
    mean_peso = df['peso'].mean()
    var_peso = df['peso'].var()
    mean_carga = df['carga'].mean()
    var_carga = df['carga'].var()
    mean_vo2 = df['vo2'].mean()
    var_vo2 = df['vo2'].var()
    print(f'Idade\nMédia: {mean_idade} | Variâcia: {var_idade}\nPeso\nMédia: {mean_peso} | Variâcia: {var_peso}\nCarga'
          f'\nMédia: {mean_carga} | Variâcia: {var_carga}\nVO2\nMédia: {mean_vo2} | Variâcia: {var_vo2}\n')


def boxplot(df):
    titles = ['Idade', 'Peso', 'Carga', 'VO2']

    f, a = plt.subplots(2, 2)
    a = a.ravel()
    for idx, ax in enumerate(a):
        ax.boxplot(df.iloc[:, idx])
        ax.set_title(titles[idx])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
