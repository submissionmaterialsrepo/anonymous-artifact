import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import csv

# Caminho para o arquivo compactado
CAMINHO_ARQUIVO = '../dados/mh1m_balanceadas.npz'

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

print(dados.files)
# Extração dos arrays principais
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

print(f"X é um {type(X)} (Shape:  {X.shape})")
print(f"y é um {type(y)} (Shape: {y.shape})")
print(f"colunas é um {type(colunas)} (Shape: ({colunas.shape})")

print(f"O tipo de dados dentro de X é: {X.dtype}")


df = pd.DataFrame(X, columns=colunas)
df['classes'] = y
print(df.shape)

del X, y, dados
gc.collect()

print(df.head(3))


print(df['classes'].value_counts())

colunas_intents = [col for col in df.columns if col.startswith('intents::')]
colunas_permissions = [col for col in df.columns if col.startswith('permissions::')]
colunas_opcodes = [col for col in df.columns if col.startswith('opcodes::')]
colunas_apicalls = [col for col in df.columns if col.startswith('apicalls::')]

df_aux = df[colunas_apicalls + ['classes']].copy()

nomes_resumidos = []
for coluna in colunas_apicalls:
  nomes_resumidos.append(coluna.split(".")[0])
nomes_resumidos = list(set(nomes_resumidos))

for nome_resumido in tqdm(nomes_resumidos):
  df[f"g_{nome_resumido}"] = df_aux.filter(like=nome_resumido).max(axis=1)

df = df.drop(columns=colunas_apicalls)


print(f"Número de colunas intents: {len(colunas_intents)}")
print(f"Número de colunas permissions: {len(colunas_permissions)}")
print(f"Número de colunas opcodes: {len(colunas_opcodes)}")
print(f"Número de colunas apicalls: {len(colunas_apicalls)}")

print(f"Número de colunas total inicial: {len(colunas_intents) + len(colunas_permissions) + len(colunas_opcodes) + len(colunas_apicalls)}")
print(f"Número de nomes resumidos: {len(nomes_resumidos)}")

print("DataFrame final (incluindo classes):")
print(df.shape)

colunas_intents = [col for col in df.columns if col.startswith('intents::')]
colunas_permissions = [col for col in df.columns if col.startswith('permissions::')]
colunas_opcodes = [col for col in df.columns if col.startswith('opcodes::')]
colunas_apicalls = [col for col in df.columns if col.startswith('g_apicalls::')]

print(f"Número de colunas intents: {len(colunas_intents)}")
print(f"Número de colunas permissions: {len(colunas_permissions)}")
print(f"Número de colunas opcodes: {len(colunas_opcodes)}")
print(f"Número de colunas apicalls: {len(colunas_apicalls)}")

colunas_intents = [col for col in df.columns if col.startswith('g_intents::')]
colunas_permissions = [col for col in df.columns if col.startswith('g_permissions::')]
colunas_opcodes = [col for col in df.columns if col.startswith('g_opcodes::')]
colunas_apicalls = [col for col in df.columns if col.startswith('g_apicalls::')]

print(f"Número de colunas intents: {len(colunas_intents)}")
print(f"Número de colunas permissions: {len(colunas_permissions)}")
print(f"Número de colunas opcodes: {len(colunas_opcodes)}")
print(f"Número de colunas apicalls: {len(colunas_apicalls)}")

X = df.drop(columns=['classes']).values.astype(np.int8)
y = df['classes'].astype(np.int8).values
colunas = df.columns.drop('classes').values
del df
gc.collect()

print(type(X), X.shape)
print(type(y), y.shape)
print(type(colunas), colunas.shape)


CAMINHO_SAIDA = '../dados/mh1m_balanceadas_apicalls_agrupadas.npz'

np.savez_compressed(CAMINHO_SAIDA, data=X, classes=y, column_names=colunas)

print(f"DataFrame salvo em {CAMINHO_SAIDA}")
