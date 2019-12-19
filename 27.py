import pandas as pd

N = 10


# df = pd.read_csv('data.csv', sep=',')
# prior = []
# low = 30.00
# high = 70.20
# hypothesis = []
# likelihood_menos_35 = []
# likelihood_mais_35 = []
#
# for i in range(6):
#     hypothesis.append((round(low, 1), round(high, 1)))
#     count_prior = df[(df['carga'] >= low) & (df['carga'] < high)]
#     prior.append(count_prior['carga'].count() / df['carga'].count())
#
#     count_likelihood_menos_35 = df[(df['carga'] >= low) & (df['carga'] < high) & (df['vo2'] < 35.0)]['vo2'].count()
#     likelihood_menos_35.append(count_likelihood_menos_35["VO2 maximo"].count() / count_prior["Carga Final"].count())
#
#     count_likelihood_mais_35 = Data[
#         (Data["Carga Final"] >= low) & (Data["Carga Final"] < high) & (Data["VO2 maximo"] >= 35.0)]
#     likelihood_mais_35.append(count_likelihood_mais_35["VO2 maximo"].count() / count_prior["Carga Final"].count())
#
#     low += 40.2
#     high += 40.2
#
# hypothesis.append((round(low, 1), round(high, 1)))
#
# count_prior = Data[(Data["Carga Final"] >= low) & (Data["Carga Final"] <= high)]
# prior.append(count_prior["Carga Final"].count() / Data["Carga Final"].count())
#
# count_likelihood_menos_35 = Data[
#     (Data["Carga Final"] >= low) & (Data["Carga Final"] <= high) & (Data["VO2 maximo"] < 35.0)]
# likelihood_menos_35.append(count_likelihood_menos_35["VO2 maximo"].count() / count_prior["Carga Final"].count())
#
# count_likelihood_mais_35 = Data[
#     (Data["Carga Final"] >= low) & (Data["Carga Final"] <= high) & (Data["VO2 maximo"] >= 35.0)]
# likelihood_mais_35.append(count_likelihood_mais_35["VO2 maximo"].count() / count_prior["Carga Final"].count())
#
# bayes_numerator_menos_35 = []
# bayes_numerator_mais_35 = []
# for i in range(0, 10):
#     bayes_numerator_menos_35.append(prior[i] * likelihood_menos_35[i])
#     bayes_numerator_mais_35.append(prior[i] * likelihood_mais_35[i])
#
# total_menos_35 = np.sum(bayes_numerator_menos_35)
# total_mais_35 = np.sum(bayes_numerator_mais_35)
#
# posterior_menos_35 = []
# posterior_mais_35 = []
#
# for i in range(0, 10):
#     posterior_menos_35.append(bayes_numerator_menos_35[i] / total_menos_35)
#     posterior_mais_35.append(bayes_numerator_mais_35[i] / total_mais_35)
#
# predict = []
# for i in range(0, 10):
#     predict.append(posterior_menos_35[i] * likelihood_mais_35[i])
#
# prob = np.sum(predict)
#
# inference_menos_35 = pd.DataFrame({"hypothesis": hypothesis,
#                                    "prior": prior,
#                                    "likelihood": likelihood_menos_35,
#                                    "Bayes Num": bayes_numerator_menos_35,
#                                    "Posterior": posterior_menos_35})
#
# inference_mais_35 = pd.DataFrame({"hypothesis": hypothesis,
#                                   "prior": prior,
#                                   "likelihood": likelihood_mais_35,
#                                   "Bayes Num": bayes_numerator_mais_35,
#                                   "Posterior": posterior_mais_35})
#
# inference_cond = pd.DataFrame({"hypothesis": hypothesis,
#                                "prior": prior,
#                                "likelihood(<35)": likelihood_menos_35,
#                                "Bayes Num1": bayes_numerator_mais_35,
#                                "Posterior 1": posterior_menos_35,
#                                "likelihood(>=35)": likelihood_mais_35,
#                                " prediction": predict})
#
# print("hipotese que VO2 máximo esta abaixo da média 35\n\n", inference_menos_35, "\n")
# print(total_menos_35, "\n\n\n\n")
# print("hipotese que VO2 máximo esta acima da média 35\n\n", inference_mais_35, "\n")
# print(total_mais_35, "\n\n\n\n")
# print("hipotese que VO2 máximo esta acima da média 35, dado que antes estava abaixo da media \n\n", inference_cond,
#       "\n")
# print(total_menos_35)
# print("prob: ", prob, "\n\n\n\n")


def main():
    df = pd.read_csv('data.csv', sep=',')

    table_35up = pd.DataFrame([['', 0, 0, 0, 0]] * N,
                              columns=['hypotesis', 'prior', 'likelihood', 'bayes_numerator', 'posterior'])
    table_35down = pd.DataFrame([['', 0, 0, 0, 0]] * N,
                                columns=['hypotesis', 'prior', 'likelihood', 'bayes_numerator', 'posterior'])
    low = df['carga'].min()
    high = low + df['carga'].max() / N

    for i in range(N):
        # Get intervall
        interval = df[(df['carga'] >= low) & (df['carga'] < high)]

        # Fill hypotesis
        table_35up.iloc[i, 0] = str((round(low, 1), round(high, 1)))
        table_35down.iloc[i, 0] = str((round(low, 1), round(high, 1)))

        # Fill prior
        table_35up.iloc[i, 1] = interval['carga'].count() / df['carga'].count()
        table_35down.iloc[i, 1] = interval['carga'].count() / df['carga'].count()

        # Fill likelihood
        table_35up.iloc[i, 2] = df[(df['carga'] >= low) & (df['carga'] < high) & (df['vo2'] >= 35.0)][
                                    'vo2'].count() / interval['carga'].count()
        table_35down.iloc[i, 2] = df[(df['carga'] >= low) & (df['carga'] < high) & (df['vo2'] < 35.0)][
                                      'vo2'].count() / interval['carga'].count()

        low += df['carga'].max() / N
        high += df['carga'].max() / N

    # Fill bayes numerator
    table_35up['bayes_numerator'] = table_35up['prior'] * table_35up['likelihood']
    table_35down['bayes_numerator'] = table_35down['prior'] * table_35down['likelihood']

    # Fill posterior
    table_35up['posterior'] = table_35up['bayes_numerator'] / table_35up['bayes_numerator'].sum()
    table_35down['posterior'] = table_35down['bayes_numerator'] / table_35down['bayes_numerator'].sum()

    conditional_prob = (table_35down['posterior'] * table_35up['likelihood']).sum()

    print(table_35up)
    print(table_35down)
    print(f'\nP(VO2 >= 35 | VO2 < 35) = {conditional_prob}')


if __name__ == '__main__':
    main()
