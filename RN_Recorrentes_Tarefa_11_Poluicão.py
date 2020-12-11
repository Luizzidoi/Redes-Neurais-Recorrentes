"""
Redes Neurais Recorrentes
Base de dados da poluição da China
Construição de uma série temporal para prever a poluição na China em horas específicas
Geração de um gráfico para comparação entre a poluição real e o previsto pela rede neural

"""



import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM    # LSTM tipo de rede neural utilizada, uma das mais eficientes
from sklearn.preprocessing import MinMaxScaler   # Normalização dos valores para valores entre 0 e 1
import numpy as np
import matplotlib.pyplot as plt
# Classe EarlyStopping: Parar o processamento antes, de acordo com algumas condições
# Classe ReduceLROnPlateau: Reduzir uma taxa de aprendizagem quando uma métrica parou de melhorar
# Classe ModelCheckpoint: Salvar o modelo depois de cada uma das épocas (gravação os pesos)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Carregando as variáveis com os atributos da base de dados """

base = pd.read_csv('poluicao.csv')
base = base.dropna()

# Exclusão de atributos que não serão utilizados para a análise
base = base.drop('No', axis=1)
base = base.drop('year', axis=1)
base = base.drop('month', axis=1)
base = base.drop('day', axis=1)
base = base.drop('hour', axis=1)
base = base.drop('cbwd', axis=1)

# Completar a base de treinamentos com todos os valores, menos o pm2.5 que será a saída
base_treinamento = base.iloc[:, 1:7].values

# Busca dos valores que será feita a previsão, ou seja o primeiro atributo pm2.5
poluicao = base.iloc[:, 0].values

# Normalização dos valores para valores entre 0 e 1. Minimizar o processamento
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
# Necessário mudar o formato da variável para pode aplicar a normalização
poluicao = poluicao.reshape(-1, 1)
poluicao_normalizado = normalizador.fit_transform(poluicao)


""" Preenchimento das variáveis com 10 datas anteriores para o treinamento """

previsores = []
poluicao_real = []
# 90 valores anteriores para previsores e 1242 o tamanho da base de dados
for i in range(10, 41757):
    previsores.append(base_treinamento_normalizada[i-10:i, 0:6])
    poluicao_real.append(poluicao_normalizado[i, 0])

# Transformação dos dados para uma tabela
previsores, poluicao_real = np.array(previsores), np.array(poluicao_real)


########################################################################################################################
""" Estrutura da rede neural recorrente """

# 6 atributos previsores
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1, activation='linear'))

# Optimizer 'rsmprop' mais utilizado para as redes neurais recorrentes
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(previsores, poluicao_real, epochs=100, batch_size=64)


# Neste exemplo não utilizou-se uma base de dados específica para teste, ou seja,
# farei as previsões diretamente na base de dados de treinamento
previsoes = regressor.predict(previsores)
previsoes = normalizador.inverse_transform(previsoes)

# Verificação da média nos resultados das previsões e nos resultados reais
print(previsoes.mean())
print(poluicao.mean())
diferenca = previsoes.mean() - poluicao.mean()
print(diferenca)


########################################################################################################################
""" Geração do gráfico. Será gerado um gráfico de barras porque temos muitos registros """

plt.plot(poluicao, color='red', label='Poluição real')
plt.plot(previsoes, color='blue', label='Previsões')
plt.title('Previsão poluição')
plt.xlabel('Horas')
plt.ylabel('Valor poluição')
plt.legend()
plt.show()

print("Fim")