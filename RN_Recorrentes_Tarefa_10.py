"""
Redes Neurais Recorrentes
Base de dados da Petrobras
Previsão do preços das ações
Geração de um gráfico para comparação entre o preço real e o previsto pela rede neural

"""


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM     # LSTM tipo de rede neural utilizada, uma das mais eficientes
from sklearn.preprocessing import MinMaxScaler    # Normalização dos valores para valores entre 0 e 1
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Carregando as veriáveis com os atributos da base de dados  """

base = pd.read_csv('petr4_treinamento_ex.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 1:2].values

# Normalização dos valores para valores entre 0 e 1. Minimizar o processamento
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)


""" Preenchimento das variáveis com 90 datas anteriores para o treinamento """

previsores = []
preco_real = []
# 90 valores anteriores para previsores e 1242 o tamanho da base de dados
for i in range(90, 1342):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])

# Transformação dos dados para uma tabela
previsores, preco_real = np.array(previsores), np.array(preco_real)

# Transformação nos dados para que o Tensorflow consiga fazer a leitura. Input shape (batch_size, timesteps, input_dim)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))


########################################################################################################################
""" Estrutura da rede neural recorrente """

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 1)))
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

regressor.fit(previsores, preco_real, epochs=100, batch_size=32)


########################################################################################################################
""" Utilizando a base de dados de teste para melhores resultados """

base_teste = pd.read_csv('petr4_teste_ex.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
# Concatenar a base de dados de treinamento com a base de dados de teste para conseguir buscar/utilizar os 90 valors anteriores.
# Axis=0 para concatenção por coluna
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
# Buscar 90 valores anteriores
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
# Deixar no formato do numpy
entradas = entradas.reshape(-1, 1)
# Normalização das entradas
entradas = normalizador.transform(entradas)

""" Utilizando a base de teste """

X_teste = []
# 90 é o início da base de teste e 112 o fim
for i in range(90, 109):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

diferenca = previsoes.mean() - preco_real_teste.mean()
print(diferenca)


########################################################################################################################
""" Geração do gráfico para análise """

plt.plot(preco_real_teste, color='red', label='Preço Real')
plt.plot(previsoes, color='blue', label='Previsões')
plt.title('Previsão do preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()