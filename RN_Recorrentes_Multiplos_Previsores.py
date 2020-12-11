"""
Redes Neurais Recorrentes
Multiplos previsores
Base de dados Bolsa de valores
Previsão do preços de ações
Geração de um gráfico para comparação entre o preço real e o previsto pela rede neural

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

base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 1:7].values

# Normalização dos valores para valores entre 0 e 1. Minimizar o processamento
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# Criação de outro normalizador utilizado nas previsões
normalizador_previsao = MinMaxScaler(feature_range=(0, 1))
normalizador_previsao.fit_transform(base_treinamento[:, 0:1])


""" Preenchimento das variáveis com 90 datas anteriores para o treinamento """

previsores = []
preco_real = []
# 90 valores anteriores para previsores e 1242 o tamanho da base de dados
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0:6])
    preco_real.append(base_treinamento_normalizada[i, 0])

# Transformação dos dados para uma tabela
previsores, preco_real = np.array(previsores), np.array(preco_real)


########################################################################################################################
""" Estrutura da rede neural recorrente """

# 6 atributos previsores
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='sigmoid'))

# Optimizer 'rsmprop' mais utilizado para as redes neurais recorrentes
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='pesos.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, preco_real, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])


########################################################################################################################
""" Utilizando a base de dados de teste para melhores resultados """

base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
frames = [base, base_teste]
base_completa = pd.concat(frames)
base_completa = base_completa.drop('Date', axis=1)

# Buscar 90 valores anteriores
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = normalizador.transform(entradas)

X_teste = []
# 90 é o início da base de teste e 112 o fim
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)

previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

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
