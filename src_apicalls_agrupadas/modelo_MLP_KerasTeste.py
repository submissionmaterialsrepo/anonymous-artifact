import os
import gc
from datetime import datetime
from math import trunc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier





def cria_CM_Macro(nome_grupo, pasta_cm, modelo_nome, y_true_all_folds, y_pred_all_folds):


    y_true_all = np.concatenate(y_true_all_folds)
    y_pred_all = np.concatenate(y_pred_all_folds)

    cm_macro = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])  # soma/aggregado dos folds
    cm_macro_norm = cm_macro.astype(float) / cm_macro.sum(axis=1, keepdims=True)  # normalizada por linha

    # salvar CSVs macro
    base_macro = f"{nome_grupo}__{modelo_nome}__MACRO"
    caminho_cm_macro_csv = os.path.join(pasta_cm, f"{base_macro}.csv")
    caminho_cm_macro_norm_csv = os.path.join(pasta_cm, f"{base_macro}__NORMALIZADA.csv")

    pd.DataFrame(cm_macro, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(caminho_cm_macro_csv, index=True)
    pd.DataFrame(cm_macro_norm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(caminho_cm_macro_norm_csv, index=True)

    # salvar figuras macro (absoluta e normalizada)
    # absoluta
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ConfusionMatrixDisplay(confusion_matrix=cm_macro, display_labels=[0, 1]).plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title(f"{nome_grupo} - {modelo_nome} - MACRO (absoluta)")
    fig.tight_layout()
    caminho_cm_macro_png = os.path.join(pasta_cm, f"{base_macro}.png")
    fig.savefig(caminho_cm_macro_png)
    plt.close(fig)

    # normalizada por linha
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ConfusionMatrixDisplay.from_predictions(
            y_true_all, y_pred_all, display_labels=[0, 1],
            normalize='true', values_format='.2f', ax=ax, colorbar=False
        )
    ax.set_title(f"{nome_grupo} - {modelo_nome} - MACRO (normalizada)")
    fig.tight_layout()
    caminho_cm_macro_norm_png = os.path.join(pasta_cm, f"{base_macro}__NORMALIZADA.png")
    fig.savefig(caminho_cm_macro_norm_png)
    plt.close(fig)
    return caminho_cm_macro_csv,caminho_cm_macro_norm_csv,caminho_cm_macro_png,caminho_cm_macro_norm_png

def cria_CM_modelo(nome_grupo, pasta_cm, modelo_nome, fold, y_test, nome_base, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Salvar CSV da matriz de confusão do experimento (fold)
    caminho_cm_csv = os.path.join(pasta_cm, f"{nome_base}.csv")
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(caminho_cm_csv, index=True)

    # Salvar figura da matriz de confusão do experimento
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benigno', 'maligno'])
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title(f"{nome_grupo} - {modelo_nome} - Fold {fold}")
    fig.tight_layout()
    caminho_cm_png = os.path.join(pasta_cm, f"{nome_base}.png")
    fig.savefig(caminho_cm_png)
    plt.close(fig)
    return caminho_cm_csv,caminho_cm_png



def definir_mlp_keras(input_dim):
    if input_dim >= 22000:
        units = [11000, 6000, 3000, 1500, 512]
    elif input_dim >= 10000:
        units = [6000, 3000, 1500, 512, 256]
    elif input_dim >= 5000:
        units = [3000, 1500, 512, 256, 128]
    elif input_dim >= 1000:
        units = [500, 512, 256, 128]
    elif input_dim > 400:
        units = [200, 100, 50]
    else:
        units = [50, 25]

    model = Sequential()
    model.add(Dense(units[0], input_shape=(input_dim,)))
    model.add(LeakyReLU(alpha=0.01))
    for u in units[1:]:
        model.add(Dense(u))
        model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model






def avaliar_modelos_em_dataframe(df, modelo_nome, nome_grupo, threshold, pasta_saida, n_splits=5):

    X = df.drop(columns=['classe']).values.astype(np.float32)
    y = df['classe'].astype(np.float32).values
    input_dim = X.shape[1]

    print(f"\nIniciando avaliação para o grupo de features: '{nome_grupo}' com {X.shape[1]} atributos e {X.shape[0]} instâncias.")

    # Pasta para matrizes de confusão
    pasta_cm = os.path.join(pasta_saida, "matrizes_confusao")
    os.makedirs(pasta_cm, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    resultados = []

    # MLP Keras
    print(f"\nIniciando MLP (Keras) para o grupo: {nome_grupo} com arquitetura dinâmica.")

    y_true_all_folds = []
    y_pred_all_folds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Treinando MLP (Keras) - Fold {fold}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        nome_base = f"{nome_grupo}__{modelo_nome}__fold{fold}"

        print(f"\nDefinindo o modelo MLP Keras.")
        model = definir_mlp_keras(input_dim)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        with tf.device('/GPU:0'):
          model.fit(X_train, y_train, epochs=30, batch_size=256, verbose=1, validation_split=0.1, callbacks=[early_stopping])
          model.save(os.path.join(pasta_saida,nome_base+'.keras'))

        y_pred_prob = model.predict(X_test).flatten()
    
        t = threshold

        y_pred = (y_pred_prob >= t).astype(int)

        # acumula para a matriz macro
        y_true_all_folds.append(y_test)
        y_pred_all_folds.append(y_pred)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test.astype(int), y_pred.astype(int), output_dict=True, zero_division=0)
        print(report)

        # ===== MATRIZ DE CONFUSÃO (2x2 com labels fixos) =====
        caminho_cm_csv, caminho_cm_png = cria_CM_modelo(nome_grupo, pasta_cm, modelo_nome, fold, y_test, nome_base, y_pred)

        for classe in ['0', '1']:
            resultados.append({
                'grupo_de_features': nome_grupo,
                'modelo': f"MLP_Keras_thr{t}",
                'classe': classe,
                'precision': report[classe]['precision'],
                'recall': report[classe]['recall'],
                'f1_score': report[classe]['f1-score'],
                'support': report[classe]['support'],
                'fold': fold,
                'accuracy_geral': acc,
                # opcional: caminho dos artefatos do fold
                'cm_csv': caminho_cm_csv if classe == '0' else '',
                'cm_png': caminho_cm_png if classe == '0' else ''
            })
        print(f"→ Threshold {t:.1f} | Fold {fold} | Acc: {acc:.4f}")

    print(f"\nAvaliação concluída para o grupo: {nome_grupo}")
    return pd.DataFrame(resultados)






DIRETORIO_BASE = "."
# Caminho para o arquivo compactado
CAMINHO_ARQUIVO = f'{DIRETORIO_BASE}/dados/mh1m_balanceadas_apicalls_agrupadas.npz'

# Carrega os dados com mmap_mode para uso mais leve de memória
dados = np.load(CAMINHO_ARQUIVO, allow_pickle=True)

# Extração dos arrays principais
X = dados['data']
y = dados['classes']
colunas = dados['column_names']

# Embaralhar X e y
rng = np.random.default_rng(42)  # garante reprodutibilidade
idx_final = rng.permutation(X.shape[0])  # embaralha os índices

X = X[idx_final]
y = y[idx_final]

print(f"Dados embaralhados: X={X.shape}, y={y.shape}")

modelos = ["mlp"]
threshold = 0.5
grupos = ["apicalls", "permissions_apicalls", "opcodes_apicalls", "intents_apicalls","todas"]



CAMINHO_RAIZ = DIRETORIO_BASE
for modelo_nome in modelos:

    for nome_grupo in grupos:

        pasta_saida = os.path.join(CAMINHO_RAIZ,nome_grupo,f"{modelo_nome}_limiar{trunc(threshold*10)}","resultados")
        os.makedirs(pasta_saida, exist_ok=True)

        # Parte 3 - Separar as colunas das features e criar os DataFrames
        # Identificar colunas por namespace ("intents, permissions, opcodes, apicalls")
        if nome_grupo == "apicalls":
            idx_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("g_apicalls::")]
            df = pd.DataFrame(X[:, idx_apicalls], columns=np.array(colunas)[idx_apicalls])
            df['classe'] = y
        elif nome_grupo == "todas":
            idx_features = range(len(colunas))
            df = pd.DataFrame(X, columns=np.array(colunas)[idx_features])
            df['classe'] = y
        elif nome_grupo == "permissions_apicalls":
            idx_permissions = [i for i, nome in enumerate(colunas) if nome.startswith("permissions::")]
            idx_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("g_apicalls::")]
            idx_features = idx_permissions + idx_apicalls
            df = pd.DataFrame(X[:, idx_features], columns=np.array(colunas)[idx_features])
            df['classe'] = y
        elif nome_grupo == "opcodes_apicalls":
            idx_opcodes = [i for i, nome in enumerate(colunas) if nome.startswith("opcodes::")]
            idx_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("g_apicalls::")]
            idx_features = idx_opcodes + idx_apicalls
            df = pd.DataFrame(X[:, idx_features], columns=np.array(colunas)[idx_features])
            df['classe'] = y
        elif nome_grupo == "intents_apicalls":
            idx_intents = [i for i, nome in enumerate(colunas) if nome.startswith("intents::")]
            idx_apicalls = [i for i, nome in enumerate(colunas) if nome.startswith("g_apicalls::")]
            idx_features = idx_intents + idx_apicalls
            df = pd.DataFrame(X[:, idx_features], columns=np.array(colunas)[idx_features])
            df['classe'] = y
        else:
            continue

        print("DataFrames criados:")
        print(f" - df : {df.shape}")

        # Parte 7 - Executar o modelo e recuperar os resultados para cada DataFrame
        df_resultados = pd.concat([
            avaliar_modelos_em_dataframe(df, modelo_nome, nome_grupo, threshold, pasta_saida, 5),
        ], ignore_index=True)


        # Parte 8.2 - Exportar os dados consolidados
        caminho_saida = os.path.join(pasta_saida, 'resultados_modelos.csv')
        df_resultados.to_csv(caminho_saida, index=False)

        resumo = df_resultados.groupby(['grupo_de_features', 'modelo', 'classe'])[['precision', 'recall', 'f1_score']].mean().round(4)
        resumo.to_csv(os.path.join(pasta_saida, 'resumo_resultados.csv'))

        # Parte 8.3 - Exibir e salvar resumo
        print(resumo)

        resumo.to_csv(os.path.join(pasta_saida, 'resumo_resultados.csv'))

        print(f"\nArquivos salvos em: {pasta_saida}")
        print("• CSV por experimento da matriz de confusão em: pasta 'matrizes_confusao/'")
        print("• PNG por experimento da matriz de confusão em: pasta 'matrizes_confusao/'")
