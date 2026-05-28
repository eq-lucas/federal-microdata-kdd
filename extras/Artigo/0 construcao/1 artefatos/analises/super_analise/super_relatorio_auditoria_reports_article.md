# Super relatório de auditoria dos resultados gerados em `reports/article`

## 0. Escopo da auditoria

Foram analisados os quatro pacotes enviados:

| Pacote                      |   Arquivos |   Tamanho extraído (MB) |   CSV |   Parquet |   PNG |   PDF |   TEX |   LATEX |
|:----------------------------|-----------:|------------------------:|------:|----------:|------:|------:|------:|--------:|
| appendix                    |        576 |                 85.0000 |    91 |         0 |   152 |   152 |    91 |      90 |
| figures                     |        284 |                164.2800 |     0 |         0 |   142 |   142 |     0 |       0 |
| tables_sem_taxas_e_sem_tree |         57 |                417.6100 |    18 |         9 |     8 |     8 |     7 |       7 |
| tree_e_taxas                |        172 |                983.1400 |    50 |        24 |    24 |    24 |    26 |      24 |

Resumo por extensão no conjunto consolidado:

| Extensão   |   Arquivos |
|:-----------|-----------:|
| .csv       |        159 |
| .latex     |        121 |
| .parquet   |         33 |
| .pdf       |        326 |
| .png       |        326 |
| .tex       |        124 |

Leituras e validações executadas:

- Todos os **159 CSVs** foram abertos com `pandas` sem erro.
- Todos os **326 PNGs** foram abertos e validados com `PIL` sem erro. Alguns heatmaps são muito grandes, mas os arquivos são válidos.
- Todos os **326 PDFs** foram abertos com `pypdf`; todos têm 1 página e nenhum apresentou erro de leitura.
- Os **33 Parquets** têm assinatura Parquet válida (`PAR1` no início e no fim). O conteúdo interno dos Parquets não foi decodificado porque o ambiente desta sessão não tem `pyarrow`/`fastparquet`; os CSVs correspondentes foram lidos e usados como base da auditoria substantiva.
- Não há arquivos `.tif` ou `.tiff` nos pacotes enviados.

Inventário detalhado dos CSVs gerado à parte: `inventario_csv_reports_article.csv`.

## 1. Cobertura dos artefatos esperados no artigo

| Artefato   | Existe   | Arquivo verificado                                                                                                                                                                |
|:-----------|:---------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Figura 1   | sim      | figures/figures/fluxo_selecao/funil/figura_1_funil_fies_inscritos_area_cine.png                                                                                                   |
| Figura 2   | sim      | figures/figures/taxas_conversao/taxa_conversao_inscritos/grafico_taxa_conversao_inscritos_saude_e_bem_estar.png                                                                   |
| Tabela 1   | sim      | tables_sem_taxas_e_sem_tree/secao_4_2/tabela_1_distribuicao_inscricoes_por_situacao.csv                                                                                           |
| Figura 3   | sim      | figures/figures/heatmap_contratados_x_nao_contratados/nacional_contratados_nao_contratados.png                                                                                    |
| Figura 4   | sim      | figures/figures/financiamento_coparticipacao/figura_4_associacao_renda_percentual_financiamento.png                                                                               |
| Tabela 2   | sim      | tables_sem_taxas_e_sem_tree/secao_4_3/tabela_2_associacao_renda_financiamento.csv                                                                                                 |
| Figura 5   | sim      | figures/figures/logit_binario/figura_probabilidade_prevista_contratacao_logit_binario_geral_holdout_80_20_e5.png                                                                  |
| Tabela 3   | sim      | tables_sem_taxas_e_sem_tree/secao_4_4/tabela_logit_binario_principal_geral_holdout_80_20_e5.csv                                                                                   |
| Figura A1  | sim      | figures/figures/fluxo_selecao/funil/figura_1a_funil_fies_candidatos_unicos_area_cine.png                                                                                          |
| Figura A2  | sim      | figures/figures/taxas_conversao/taxa_conversao_curso_priorizado/grafico_taxa_conversao_curso_priorizado_saude_e_bem_estar.png                                                     |
| Figura A3  | sim      | figures/figures/fluxo_selecao/funil/funil_fies_inscritos_regiao_total.png                                                                                                         |
| Tabela B1  | sim      | appendix/appendix/apendice_b/tabela_b1_distribuicao_inscricoes_por_faixa_renda.csv                                                                                                |
| Figura B1  | sim      | appendix/appendix/apendice_b/heatmap_lista_espera/figura_b1_lista_espera.png                                                                                                      |
| Figura C1  | sim      | appendix/appendix/apendice_modelagem/logit_binario/holdout_80_20/geral/figuras_curvas/figura_C1_probabilidade_por_desempenho_faixa_renda_logit_binario_geral_holdout_80_20_e5.png |
| Figura C2  | sim      | appendix/appendix/apendice_modelagem/logit_binario/holdout_80_20/geral/figuras_curvas/figura_C2_probabilidade_por_renda_desempenho_logit_binario_geral_holdout_80_20_e5.png       |
| Tabela C1  | sim      | appendix/appendix/apendice_modelagem/logit_binario/holdout_80_20/geral/tabela_compacta_logit_binario_geral_holdout_80_20.csv                                                      |
| Tabela C3  | sim      | appendix/appendix/apendice_modelagem/logit_binario/holdout_80_20/geral/tabela_logit_binario_geral_holdout_80_20.csv                                                               |

Veredito: os artefatos principais do corpo do artigo e os principais artefatos dos apêndices A, B e C estão presentes. A exceção importante é a **Tabela C2** descrita no texto do apêndice, que não aparece nos pacotes enviados como tabela separada de coeficientes de maior magnitude absoluta.

## 2. Integridade estrutural dos arquivos

Pontos positivos:

- Não há CSV corrompido.
- Não há PNG corrompido.
- Não há PDF corrompido.
- Para todos os PNGs existe PDF correspondente com o mesmo stem.
- Não há resíduo `.tif/.tiff`.
- Não apareceu `nome_cine_area_especifica` em nenhum CSV/TEX/LATEX.

Pontos de atenção estrutural:

- Existem CSVs técnicos sem `.tex/.latex`, o que é normal para bases de apoio, como `dados_probabilidades_*`, matrizes em formato longo e resultados brutos de regressão.
- As figuras de heatmap por área CINE têm resolução muito alta; são válidas, mas pesadas para visualização e edição em editores de texto. Para submissão, o PDF tende a ser mais seguro que o PNG enorme.
- A expressão `nome_cine_area_geral` aparece apenas em arquivos descritivos/taxas/matrizes por área CINE, não em tabelas de modelagem. Arquivos onde aparece:

  - `tree_e_taxas/taxas_conversao/taxa_conversao_curso_priorizado_dados.csv`
  - `tree_e_taxas/taxas_conversao/taxa_conversao_inscritos_dados.csv`
  - `tables_sem_taxas_e_sem_tree/matrizes_renda_desempenho/matriz_status_por_area_cine.csv`

## 3. Resultados descritivos

### 3.1 Tabela 1 — distribuição das situações de inscrição

| Situação da inscrição   |   Quantidade de inscrições |       % |
|:------------------------|---------------------------:|--------:|
| Não contratado          |                    1024604 | 46.6315 |
| Lista de espera         |                     740698 | 33.7105 |
| Contratada              |                     160369 |  7.2987 |
| Opção não contratada    |                     139469 |  6.3475 |
| Participação cancelada  |                     111202 |  5.0610 |
| Pré-selecionado         |                      12923 |  0.5881 |
| Rejeitada pela CPSA     |                       5645 |  0.2569 |
| Inscrição postergada    |                       2324 |  0.1058 |

Total da Tabela 1: **2.197.234** inscrições; percentuais somam **100.00%**.

Leitura: as três categorias centrais para a modelagem aparecem com os maiores pesos: `Não contratado`, `Lista de espera` e `Contratada`. Isso sustenta a escolha dos targets binário e ternário.

### 3.2 Tabela B1 — distribuição por faixa de renda

| Faixa de renda per capita (R$)   |   Inscrições |       % |   % acumulado |
|:---------------------------------|-------------:|--------:|--------------:|
| 601–1.200                        |       450100 | 40.5545 |       40.5545 |
| Até 600                          |       320976 | 28.9203 |       69.4748 |
| 1.201–1.800                      |       199204 | 17.9485 |       87.4232 |
| 1.801–2.400                      |        74404 |  6.7039 |       94.1271 |
| 2.401–3.000                      |        49450 |  4.4555 |       98.5826 |
| Acima de 3.000                   |        15731 |  1.4174 |      100.0000 |

Total da Tabela B1: **1.109.865** inscrições. As três primeiras faixas acumulam **87.42%** dos registros.

Observação técnica: a matriz nacional de renda × desempenho soma 1.109.861 registros, enquanto a Tabela B1 soma 1.109.865. A diferença é de **4 registros**. Isso é desprezível substantivamente, mas indica que a Tabela B1 e a matriz provavelmente não usam exatamente a mesma regra de filtragem para `gap`/célula válida. Se o texto usar a Tabela B1 para contextualizar a matriz, a diferença não altera a interpretação.

### 3.3 Matrizes de renda e desempenho

A matriz nacional em formato longo tem 279 linhas, com as colunas `faixa_renda_bruta`, `nivel_nota_gap`, `status_final`, `qtd`, `total_celula` e `percentual_celula`.

Validação das células:

- Número de células renda × desempenho: **36**.
- Soma mínima dos percentuais por célula: **100.000000%**.
- Soma máxima dos percentuais por célula: **100.000000%**.

Distribuição dos status dentro da matriz nacional:

| status_final           |    qtd |
|:-----------------------|-------:|
| NÃO CONTRATADO         | 478771 |
| LISTA DE ESPERA        | 403649 |
| CONTRATADA             | 155651 |
| PARTICIPAÇÃO CANCELADA |  54556 |
| OPÇÃO NÃO CONTRATADA   |   6534 |
| PRÉ-SELECIONADO        |   6467 |
| REJEITADA PELA CPSA    |   2988 |
| INSCRIÇÃO POSTERGADA   |   1245 |

### 3.4 Taxas de conversão

Foram encontrados dois datasets de taxas, ambos com 368 linhas:

- `taxa_conversao_inscritos_dados.csv`
- `taxa_conversao_curso_priorizado_dados.csv`

Cobertura: anos 2019–2021, semestres 1 e 2, 7 grupos regionais e 12 categorias de área/total. Há 2 valores `NaN` em cada dataset, ambos em `CINE Não Informado (MEC)` com denominador zero; isso não é erro analítico.

Saúde e bem-estar — Média Nacional, conversão por inscrições:

| periodo   |   vagas_ocupadas |   Inscritos_Geral |   taxa_conversao_inscritos |
|:----------|-----------------:|------------------:|---------------------------:|
| '19.1     |            19656 |       290709.0000 |                     6.7614 |
| '19.2     |            11676 |       123201.0000 |                     9.4772 |
| '20.1     |            15230 |       251464.0000 |                     6.0565 |
| '20.2     |             7291 |       111145.0000 |                     6.5599 |
| '21.1     |            11340 |       140037.0000 |                     8.0979 |
| '21.2     |            13253 |       133358.0000 |                     9.9379 |

Saúde e bem-estar — Média Nacional, conversão por curso priorizado:

| periodo   |   vagas_ocupadas |   Candidatos_Unicos_Geral |   taxa_conversao_curso_priorizado |
|:----------|-----------------:|--------------------------:|----------------------------------:|
| '19.1     |            19656 |               148744.0000 |                           13.2147 |
| '19.2     |            11676 |                67485.0000 |                           17.3016 |
| '20.1     |            15230 |               130115.0000 |                           11.7050 |
| '20.2     |             7291 |                61423.0000 |                           11.8701 |
| '21.1     |            11340 |                76200.0000 |                           14.8819 |
| '21.2     |            13253 |                72144.0000 |                           18.3702 |

## 4. Renda, financiamento e coparticipação

Tabela 2 gerada:

| Medida                          | Valor      |
|:--------------------------------|:-----------|
| Observações válidas             | 160.323    |
| Correlação de Pearson           | -0,7057    |
| p-valor da correlação           | < 0,001    |
| Intercepto                      | 91,62      |
| Coeficiente da renda per capita | -0,0188    |
| Variação estimada a cada R$ 100 | -1,88 p,p, |
| R²                              | 0,4980     |

Resultados brutos da regressão/correlação:

|             n |   pearson_r |   pearson_p_value |   intercepto |   beta_renda |   efeito_100_reais |       r2 |   regressao_p_value |   erro_padrao_beta |   erro_padrao_intercepto |
|--------------:|------------:|------------------:|-------------:|-------------:|-------------------:|---------:|--------------------:|-------------------:|-------------------------:|
| 160323.000000 |   -0.705712 |          0.000000 |    91.617253 |    -0.018760 |          -1.875977 | 0.498029 |            0.000000 |           0.000047 |                 0.046996 |

Leitura: a associação negativa entre renda per capita e percentual financiado está consistente. O resultado atual é `r = -0,7057`, `beta = -0,01876` e `R² = 0,4980`, com 160.323 observações válidas.

Ponto de atualização para o texto do artigo: se o manuscrito ainda trouxer `160.317`, `r = -0,7062` ou `R² = 0,4987`, esses valores pertencem a uma execução anterior. A versão atual dos reports pede atualização para os valores acima.

## 5. Regressão logística

### 5.1 Métricas principais — E5 holdout

| Modelo                  | N ABT     | N treino   |   N teste |   ROC-AUC |   Acurácia bal. |   F1/F1 macro |   Coef. renda |   Coef. gap |   Coef. interação |   Blocos |
|:------------------------|:----------|:-----------|----------:|----------:|----------------:|--------------:|--------------:|------------:|------------------:|---------:|
| logit_binario           | 1.184.972 | 947.977    |  236.9950 |    0.8446 |          0.7727 |        0.4357 |       -0.4047 |      0.1617 |            0.0976 |       20 |
| logit_ternario_geral    | 1.925.666 | 1.540.532  |  385.1340 |    0.8971 |          0.7407 |        0.6347 |       -0.3457 |      0.8881 |           -0.0396 |       20 |
| logit_ternario_medicina | 311.329   | 249.063    |   62.2660 |    0.9685 |          0.7082 |        0.5365 |       -0.0702 |      2.4360 |           -0.1454 |       19 |

Leitura:

- No logit binário geral, o E5 tem ROC-AUC 0,8446 e acurácia balanceada 0,7727. O sinal é coerente com o argumento principal: renda negativa, gap positivo e interação positiva.
- No logit ternário geral, o desempenho da classe contratada é o coeficiente mais forte entre as variáveis centrais. A renda segue negativa, mas a interação aparece levemente negativa nesta execução.
- No logit ternário Medicina, o `gap` domina a classe contratada. Renda e interação são negativas, mas com magnitude muito menor que o desempenho.

### 5.2 Matrizes de confusão — E5 holdout

| Modelo                  | Classe          |   Suporte |   Precisão |   Recall |     F1 |
|:------------------------|:----------------|----------:|-----------:|---------:|-------:|
| logit_binario           | Não contratado  |    204921 |     0.9740 |   0.6576 | 0.7851 |
| logit_binario           | Contratada      |     32074 |     0.2887 |   0.8878 | 0.4357 |
| logit_ternario_geral    | Lista de espera |    148139 |     0.8271 |   0.8205 | 0.8238 |
| logit_ternario_geral    | Não contratado  |    204921 |     0.8802 |   0.5707 | 0.6924 |
| logit_ternario_geral    | Contratada      |     32074 |     0.2531 |   0.8311 | 0.3880 |
| logit_ternario_medicina | Lista de espera |     57517 |     0.9972 |   0.8715 | 0.9301 |
| logit_ternario_medicina | Não contratado  |      3286 |     0.3664 |   0.5435 | 0.4377 |
| logit_ternario_medicina | Contratada      |      1463 |     0.1457 |   0.7095 | 0.2418 |

Leitura:

- O binário geral captura muitos contratos reais: recall da classe `Contratada` ≈ 0,888. A precisão da classe contratada é baixa ≈ 0,289, padrão esperado em base desbalanceada com `class_weight=balanced`.
- No ternário geral, `Lista de espera` é bem separada, `Não contratado` tem boa precisão mas recall menor, e `Contratada` mantém alta sensibilidade com precisão baixa.
- Em Medicina, `Lista de espera` é muito bem identificada. A separação entre `Não contratado` e `Contratada` permanece mais difícil, o que é substantivamente plausível porque parte da contratação depende de fatores não observados no modelo.

### 5.3 Probabilidades previstas

Resumo por classe real nos principais modelos:

| model_folder                                                       |   target |   count |   mean |   median |
|:-------------------------------------------------------------------|---------:|--------:|-------:|---------:|
| secao_4_4                                                          |     0.00 |  204921 |  30.63 |    12.69 |
| secao_4_4                                                          |     1.00 |   32074 |  69.21 |    72.48 |
| secao_4_5_logit_ternario_recorte_geral                             |     0.00 |  148139 |  10.80 |     1.63 |
| secao_4_5_logit_ternario_recorte_geral                             |     1.00 |  204921 |  26.75 |     9.84 |
| secao_4_5_logit_ternario_recorte_geral                             |     2.00 |   32074 |  62.19 |    66.84 |
| secao_4_5_logit_ternario_recorte_medicina                          |     0.00 |   57517 |   8.84 |     0.40 |
| secao_4_5_logit_ternario_recorte_medicina                          |     1.00 |    3286 |  38.40 |    41.98 |
| secao_4_5_logit_ternario_recorte_medicina                          |     2.00 |    1463 |  51.98 |    54.92 |
| secao_treeClassification_14_profundidade_binario_recorte_geral     |     0.00 |  204921 |  29.96 |    16.50 |
| secao_treeClassification_14_profundidade_binario_recorte_geral     |     1.00 |   32074 |  69.47 |    73.46 |
| secao_treeClassification_14_profundidade_ternario_recorte_geral    |     0.00 |  148139 |   7.46 |     0.54 |
| secao_treeClassification_14_profundidade_ternario_recorte_geral    |     1.00 |  204921 |  27.57 |     9.80 |
| secao_treeClassification_14_profundidade_ternario_recorte_geral    |     2.00 |   32074 |  64.64 |    69.97 |
| secao_treeClassification_14_profundidade_ternario_recorte_medicina |     0.00 |   57517 |   3.71 |     0.00 |
| secao_treeClassification_14_profundidade_ternario_recorte_medicina |     1.00 |    3286 |  39.83 |    42.74 |
| secao_treeClassification_14_profundidade_ternario_recorte_medicina |     2.00 |    1463 |  55.10 |    62.08 |

O ordenamento das probabilidades é coerente: as médias de `prob_contratacao` são maiores entre registros efetivamente contratados do que entre não contratados/lista de espera.

Valores atuais relevantes da Figura 5, calculados a partir de `dados_probabilidades_logit_binario_geral_holdout_80_20_e5.csv`:

| Faixa   | Desempenho   |   Prob. média atual (%) |
|:--------|:-------------|------------------------:|
| I       | < -150       |                   37.05 |
| I       | -150 a -50   |                   36.54 |
| I       | > +150       |                   59.46 |
| V       | < -150       |                   12.00 |
| V       | > +150       |                   42.21 |
| VI      | < -150       |                    9.69 |
| VI      | > +150       |                   42.71 |

Ponto de atualização para o texto: os valores mencionados no manuscrito para a Figura 5 precisam ser confrontados com essa execução. Por exemplo, na faixa I, `> +150` está em 59,46%; na faixa V, `< -150` está em 12,00% e `> +150` em 42,21%. A interpretação geral permanece a mesma, mas os números pontuais mudaram.

## 6. Árvores de decisão

### 6.1 Métricas principais — E5 holdout

| Target   | Recorte   |   Prof. máx. |   ROC-AUC |   Acurácia bal. |   F1/F1 macro |   Imp. renda |   Imp. gap |   Imp. interação | 1ª divisão   |   Prof. obs. |    Folhas |   Blocos |
|:---------|:----------|-------------:|----------:|----------------:|--------------:|-------------:|-----------:|-----------------:|:-------------|-------------:|----------:|---------:|
| binario  | geral     |            6 |    0.8295 |          0.7602 |        0.4241 |       0.0559 |     0.0002 |           0.0189 | opcao_curso  |       6.0000 |   55.0000 |  20.0000 |
| binario  | geral     |           10 |    0.8421 |          0.7700 |        0.4320 |       0.0637 |     0.0029 |           0.0195 | opcao_curso  |      10.0000 |  357.0000 |  20.0000 |
| binario  | geral     |           14 |    0.8455 |          0.7720 |        0.4391 |       0.0661 |     0.0057 |           0.0206 | opcao_curso  |      14.0000 |  810.0000 |  20.0000 |
| binario  | geral     |           19 |    0.8458 |          0.7724 |        0.4386 |       0.0671 |     0.0070 |           0.0215 | opcao_curso  |      19.0000 | 1187.0000 |  20.0000 |
| ternario | geral     |            6 |    0.8716 |          0.7174 |        0.5785 |       0.0134 |     0.0330 |           0.2970 | renda_gap    |       6.0000 |   58.0000 |  20.0000 |
| ternario | geral     |           10 |    0.9080 |          0.7527 |        0.6473 |       0.0148 |     0.0652 |           0.2595 | renda_gap    |      10.0000 |  374.0000 |  20.0000 |
| ternario | geral     |           14 |    0.9174 |          0.7627 |        0.6513 |       0.0156 |     0.0666 |           0.2513 | renda_gap    |      14.0000 |  865.0000 |  20.0000 |
| ternario | geral     |           19 |    0.9203 |          0.7649 |        0.6590 |       0.0160 |     0.0669 |           0.2488 | renda_gap    |      19.0000 | 1243.0000 |  20.0000 |
| ternario | medicina  |            6 |    0.9819 |          0.7345 |        0.5862 |       0.0004 |     0.7982 |           0.0064 | gap          |       6.0000 |   53.0000 |  19.0000 |
| ternario | medicina  |           10 |    0.9846 |          0.7465 |        0.6061 |       0.0031 |     0.7702 |           0.0070 | gap          |      10.0000 |  135.0000 |  19.0000 |
| ternario | medicina  |           14 |    0.9846 |          0.7366 |        0.6027 |       0.0039 |     0.7657 |           0.0073 | gap          |      14.0000 |  161.0000 |  19.0000 |
| ternario | medicina  |           19 |    0.9846 |          0.7366 |        0.6027 |       0.0039 |     0.7656 |           0.0073 | gap          |      15.0000 |  162.0000 |  19.0000 |

Leitura por recorte:

- Binário geral: a primeira divisão é `opcao_curso` em todas as profundidades. A profundidade 14 melhora em relação à padrão 6 e quase empata com 19, com menos complexidade que 19.
- Ternário geral: a primeira divisão é `renda_gap` em todas as profundidades. Isso reforça a importância da interação entre renda e desempenho na separação entre lista de espera, não contratado e contratada.
- Ternário Medicina: a primeira divisão é `gap` em todas as profundidades. O desempenho acadêmico domina o modelo; as profundidades 14 e 19 praticamente saturam.

### 6.2 Matrizes de confusão — profundidade 14, E5 holdout

| Modelo                   | Classe          |   Suporte |   Precisão |   Recall |     F1 |
|:-------------------------|:----------------|----------:|-----------:|---------:|-------:|
| tree14_binario           | Não contratado  |    204921 |     0.9714 |   0.6703 | 0.7933 |
| tree14_binario           | Contratada      |     32074 |     0.2932 |   0.8737 | 0.4391 |
| tree14_ternario_geral    | Lista de espera |    148139 |     0.8617 |   0.8554 | 0.8585 |
| tree14_ternario_geral    | Não contratado  |    204921 |     0.8869 |   0.5717 | 0.6953 |
| tree14_ternario_geral    | Contratada      |     32074 |     0.2606 |   0.8611 | 0.4001 |
| tree14_ternario_medicina | Lista de espera |     57517 |     0.9983 |   0.9275 | 0.9616 |
| tree14_ternario_medicina | Não contratado  |      3286 |     0.3672 |   0.5666 | 0.4456 |
| tree14_ternario_medicina | Contratada      |      1463 |     0.2785 |   0.7157 | 0.4009 |

Leitura:

- A árvore 14 melhora ou aproxima os resultados do logit, mas mantém o mesmo problema estrutural: a classe `Contratada` tem recall alto e precisão baixa.
- Em Medicina, a árvore 14 é melhor que o logit na precisão da classe contratada nesta execução, mas ainda não deve ser lida como classificador individual perfeito.

### 6.3 Papel das árvores no artigo

As árvores são adequadas como análise complementar, especialmente para apoiar a discussão sobre hierarquia de variáveis:

- binário geral: importância operacional de `opcao_curso` e renda;
- ternário geral: centralidade da interação `renda_gap`;
- Medicina: centralidade quase absoluta de `gap`.

Nos pacotes enviados existem figuras de importância das variáveis para as árvores, mas não há tabelas suplementares D1/D2 separadas em CSV/TEX. Se o apêndice D for incorporado ao manuscrito, recomendo gerar tabelas explícitas de importâncias das variáveis centrais e top 5 variáveis por modelo.

## 7. Consistência das variáveis CINE/subárea

Resultado da busca textual em CSV/TEX/LATEX:

- `nome_cine_area_especifica`: **0 ocorrência**.
- `nome_cine_area_geral`: **3 ocorrências**, todas em arquivos descritivos/taxas/matrizes por área CINE.

Isso é coerente com a decisão metodológica atual: a modelagem não usa `nome_cine_area_geral` nem `nome_cine_area_especifica`; essas variáveis aparecem apenas em produtos descritivos por área CINE.

## 8. Lacunas ou pontos que precisam de decisão

1. **Tabela C2 ausente nos reports enviados.** O texto do Apêndice C menciona uma tabela de coeficientes de maior magnitude absoluta. Nos pacotes analisados, existem a tabela compacta de métricas e a tabela de especificações, mas não uma Tabela C2 separada. É necessário gerar ou remover/ajustar essa chamada no texto.

2. **Valores pontuais do texto precisam ser atualizados.** A execução atual alterou levemente os resultados da subseção 4.3 e alguns percentuais da Figura 5. A interpretação não muda, mas os números do manuscrito devem vir dos CSVs atuais.

3. **Diferença de 4 registros entre B1 e matriz nacional.** A diferença é mínima, mas vale confirmar se B1 deve usar exatamente o mesmo filtro da matriz nacional ou se é uma distribuição complementar com regra própria.

4. **Figuras de heatmap muito grandes.** Arquivos válidos, mas pesados. Para manuscrito, o PDF é preferível; para repositório, talvez seja melhor manter PNGs só quando necessário.

5. **Parquets não decodificados nesta sessão.** A assinatura Parquet está válida, e os CSVs correspondentes foram auditados. Para auditoria integral de conteúdo Parquet, seria necessário rodar em ambiente com `pyarrow` ou `fastparquet`.

## 9. Veredito final

Os resultados estão tecnicamente consistentes e completos para o corpo principal do artigo, com boa cobertura de figuras, tabelas e apêndices. Não encontrei corrupção de arquivo, ausência de formatos essenciais, nem indício de que variáveis CINE indevidas estejam nos outputs de modelagem.

A execução atual está apta para uso analítico, com três correções editoriais/metodológicas antes de fechar o manuscrito:

- atualizar os números pontuais do texto para os valores atuais dos CSVs;
- gerar ou remover a Tabela C2 do Apêndice C;
- decidir se as tabelas suplementares das árvores entram como Apêndice D e, se sim, gerar tabelas explícitas em CSV/TEX.

## 10. Resumo executivo

**OK:** CSVs, PDFs, PNGs, estruturas principais, Tabela 1, Tabela B1, Figura B1, Figura 1–5, modelos logísticos, árvores padrão e profundas, probabilidades previstas, matrizes de confusão e outputs por seção.

**Ajustar:** Tabela C2 ausente; valores do texto precisam ser sincronizados com a execução atual; pequena diferença B1 vs matriz; possível criação formal do Apêndice D para árvores.

**Não identificado:** uso de `nome_cine_area_especifica` nos outputs; TIFF restante; arquivos corrompidos; quebras de leitura em CSV/PDF/PNG.
