# Classificação de Malware Android: Estudo Experimental com a Base MH-1M e Seleção de Atributos Guiada por XAI

> Repositório de artefatos do artigo **"Classificação de Malware Android: Um Estudo sobre a Base MH-1M com Seleção de Features Guiada por XAI"**. Inclui scripts de pré-processamento, pipelines de seleção de atributos, experimentos de classificação e mecanismos de reprodutibilidade.

## Estrutura do Repositório

```text
.
├── dados/                    # Datasets e script de curadoria amostral
├── src_inicial/              # Pipeline base — dataset balanceado, sem redução
├── src_shap/                 # Pipeline XAI — seleção por valores SHAP
├── src_chi2/                 # Pipeline estatístico — teste Chi-Square (χ²)
├── src_rfe/                  # Pipeline wrapper — Recursive Feature Elimination
├── src_apicalls_agrupadas/   # Pipeline de agrupamento semântico de API calls
└── README.md
```

---

## Descrição dos Módulos

### `dados/`

Diretório destinado ao armazenamento do dataset original e do subconjunto amostral gerado para os experimentos. **Inicialmente vazio** — o dataset MH-1M deve ser obtido diretamente do repositório fonte e depositado aqui antes da execução de qualquer pipeline:

**URL:** https://github.com/Malware-Hunter/MH-1M

Também contém o script de curadoria amostral, responsável por aplicar os critérios de filtragem e balanceamento descritos no artigo, produzindo o subconjunto efetivamente utilizado nos experimentos.

---

### `src_inicial/` — Baseline sem Redução de Dimensionalidade

Pipeline de referência executado sobre o dataset balanceado integral, sem qualquer etapa de seleção de atributos. Os scripts desta pasta:

- Treinam e avaliam os modelos de classificação propostos;
- Computam métricas de desempenho para cada grupo de features;
- Criam automaticamente os diretórios de saída para armazenamento de resultados, métricas e artefatos dos experimentos.

---

### `src_shap/` — Seleção por XAI (SHAP)

Pipeline baseado em *SHapley Additive exPlanations* (SHAP) para seleção de atributos orientada por explicabilidade. Os scripts desta pasta:

- Calculam os valores de importância SHAP para cada atributo;
- Agregam as importâncias individuais por média, produzindo uma importância global por feature;
- Geram rankings ordinais de atributos com base nas importâncias consolidadas;
- Constroem datasets reduzidos a partir dos rankings obtidos;
- Retreinam e avaliam os modelos sobre os subconjuntos gerados.

---

### `src_chi2/` — Seleção por Teste Estatístico (χ²)

Pipeline baseado no teste de independência Chi-Square (χ²) para seleção de atributos por relevância estatística. Os scripts desta pasta:

- Calculam a pontuação χ² de cada atributo em relação à variável-alvo;
- Selecionam o mesmo número de features utilizado na abordagem SHAP, garantindo comparação controlada entre os métodos;
- Geram datasets reduzidos com os atributos selecionados;
- Treinam os modelos e armazenam os resultados obtidos.

---

### `src_rfe/` — Seleção por Eliminação Recursiva (RFE)

Pipeline de seleção do tipo *wrapper* via *Recursive Feature Elimination* (RFE). Os scripts desta pasta:

- Executam a eliminação recursiva de atributos com base na importância atribuída pelo estimador base;
- Mantêm paridade no número de features selecionadas em relação às demais abordagens;
- Geram datasets reduzidos com os atributos remanescentes;
- Treinam os modelos e armazenam os resultados obtidos.

---

### `src_apicalls_agrupadas/` — Agrupamento Semântico de API Calls

Pipeline dedicado ao agrupamento de features do tipo API call por categoria semântica e à construção do dataset resultante. Os scripts desta pasta:

- Implementam a lógica de agrupamento das features de API calls em categorias funcionais, consolidando chamadas semanticamente relacionadas em atributos agregados;
- Geram o dataset derivado com as features agrupadas, substituindo as API calls individuais pelas representações categóricas;
- Treinam e avaliam os modelos de classificação sobre o dataset resultante;
- Armazenam os resultados, métricas e artefatos dos experimentos nos diretórios de saída criados automaticamente.

---

## Ambiente Experimental

| Parâmetro               | Configuração                 |
|-------------------------|------------------------------|
| Plataforma              | Google Colab Pro+            |
| Acelerador              | NVIDIA A100 (GPU)            |
| Configuração de memória | High-RAM                     |
| Linguagem               | Python 3 (notebooks Jupyter) |

> Devido ao volume da base de dados e ao custo computacional dos métodos de seleção e treinamento, recomenda-se hardware equivalente para reprodução integral dos experimentos.

---

## Protocolo de Reprodução

1. Obter o dataset MH-1M no repositório fonte.
2. Depositar os arquivos em `dados/`.
3. Executar o script de curadoria e balanceamento amostral.
4. Executar o pipeline baseline (`src_inicial/`).
5. Executar o pipeline SHAP (`src_shap/`).
6. Executar o pipeline Chi² (`src_chi2/`).
7. Executar o pipeline RFE (`src_rfe/`).
8. Executar o pipeline de agrupamento de API calls (`src_apicalls_agrupadas/`).
9. Consolidar e comparar as métricas obtidas por cada abordagem.

---

## Artefatos Gerados

Cada pipeline cria automaticamente uma estrutura de diretórios contendo:

- Métricas de classificação (acurácia, F1, AUC, etc.);
- Modelos treinados serializados;
- Rankings ordinais de atributos;
- Datasets reduzidos ou derivados por método;
- Arquivos auxiliares para análise e replicação dos resultados.

---

## Licença

Disponibilizado exclusivamente para fins acadêmicos e de pesquisa. Reprodução ou uso comercial não autorizados.
