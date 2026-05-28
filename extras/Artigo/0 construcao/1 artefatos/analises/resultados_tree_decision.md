# Análise dos resultados `treeClassification`

Relatório gerado a partir do arquivo `reports(1).zip`. Foram lidos os CSVs, as matrizes de confusão, as tabelas principais, as importâncias e as figuras PNG/PDF da etapa `treeClassification`.

## Integridade geral

- Arquivos totais no `reports`: **617**.

- Arquivos `treeClassification`: **220**.

- Extensões `treeClassification`: `{'.log': 3, '.csv': 55, '.parquet': 6, '.latex': 24, '.pdf': 54, '.png': 54, '.tex': 24}`.

- CSVs `treeClassification` lidos sem erro: **55**. Erros: **0**.

- Não encontrei pastas/arquivos com `medicina_medicina`, `randomForest`, `RandomForest`, `random_forest` ou `recorte_medicinaecorte`.

- O pacote analisado não contém `.tiff`; há `.png` e `.pdf` correspondentes nas figuras principais.

## Métricas dos experimentos

### Árvore binária geral — holdout

| experimento   |   n_treino |   n_teste | primeira_divisao   |   tree_depth_observada |   tree_n_leaves |   roc_auc |   accuracy |   balanced_accuracy |     f1 |   log_loss |
|:--------------|-----------:|----------:|:-------------------|-----------------------:|----------------:|----------:|-----------:|--------------------:|-------:|-----------:|
| E1            |     947420 |    236855 | renda_gap          |                      6 |              62 |    0.6431 |     0.5389 |              0.6006 | 0.2869 |     0.6594 |
| E2            |     947420 |    236855 | opcao_curso        |                      6 |              55 |    0.8232 |     0.6387 |              0.7579 | 0.4084 |     0.5107 |
| E3            |     947420 |    236855 | opcao_curso        |                      6 |              54 |    0.8301 |     0.6611 |              0.7606 | 0.4174 |     0.5049 |
| E4            |     947420 |    236855 | opcao_curso        |                      6 |              55 |    0.8309 |     0.6652 |              0.7612 | 0.4193 |     0.5039 |
| E5            |     947420 |    236855 | opcao_curso        |                      6 |              56 |    0.8299 |     0.6863 |              0.7619 | 0.4276 |     0.5032 |

### Árvore binária geral — in-sample

| experimento   |   n_treino |   n_teste | primeira_divisao   |   tree_depth_observada |   tree_n_leaves |   roc_auc |   accuracy |   balanced_accuracy |     f1 |   log_loss |
|:--------------|-----------:|----------:|:-------------------|-----------------------:|----------------:|----------:|-----------:|--------------------:|-------:|-----------:|
| E1            |    1184275 |   1184275 | renda_gap          |                      6 |              64 |    0.6453 |     0.5371 |              0.6023 | 0.2880 |     0.6593 |
| E2            |    1184275 |   1184275 | opcao_curso        |                      6 |              55 |    0.8248 |     0.6437 |              0.7593 | 0.4108 |     0.5105 |
| E3            |    1184275 |   1184275 | opcao_curso        |                      6 |              53 |    0.8308 |     0.6609 |              0.7617 | 0.4181 |     0.5050 |
| E4            |    1184275 |   1184275 | opcao_curso        |                      6 |              54 |    0.8319 |     0.6629 |              0.7620 | 0.4189 |     0.5040 |
| E5            |    1184275 |   1184275 | opcao_curso        |                      6 |              55 |    0.8305 |     0.6819 |              0.7615 | 0.4255 |     0.5033 |

### Árvore ternária geral — holdout

| experimento   |   n_treino |   n_teste | primeira_divisao   |   tree_depth_observada |   tree_n_leaves |   roc_auc_ovr_weighted |   accuracy |   balanced_accuracy |   f1_macro |   log_loss |
|:--------------|-----------:|----------:|:-------------------|-----------------------:|----------------:|-----------------------:|-----------:|--------------------:|-----------:|-----------:|
| E1            |    1530711 |    382678 | renda_gap          |                      6 |              63 |                 0.7619 |     0.4824 |              0.5502 |     0.4323 |     0.9028 |
| E2            |    1530711 |    382678 | renda_gap          |                      6 |              59 |                 0.8263 |     0.5901 |              0.6754 |     0.5530 |     0.7329 |
| E3            |    1530711 |    382678 | renda_gap          |                      6 |              59 |                 0.8692 |     0.6101 |              0.7132 |     0.5770 |     0.6779 |
| E4            |    1530711 |    382678 | renda_gap          |                      6 |              59 |                 0.8690 |     0.6101 |              0.7132 |     0.5770 |     0.6790 |
| E5            |    1530711 |    382678 | renda_gap          |                      6 |              58 |                 0.8748 |     0.6098 |              0.7176 |     0.5761 |     0.6621 |

### Árvore ternária geral — in-sample

| experimento   |   n_treino |   n_teste | primeira_divisao   |   tree_depth_observada |   tree_n_leaves |   roc_auc_ovr_weighted |   accuracy |   balanced_accuracy |   f1_macro |   log_loss |
|:--------------|-----------:|----------:|:-------------------|-----------------------:|----------------:|-----------------------:|-----------:|--------------------:|-----------:|-----------:|
| E1            |    1913389 |   1913389 | renda_gap          |                      6 |              63 |                 0.7612 |     0.4826 |              0.5521 |     0.4329 |     0.9028 |
| E2            |    1913389 |   1913389 | renda_gap          |                      6 |              59 |                 0.8260 |     0.5803 |              0.6746 |     0.5420 |     0.7328 |
| E3            |    1913389 |   1913389 | renda_gap          |                      6 |              59 |                 0.8690 |     0.6099 |              0.7132 |     0.5770 |     0.6782 |
| E4            |    1913389 |   1913389 | renda_gap          |                      6 |              59 |                 0.8687 |     0.6099 |              0.7132 |     0.5770 |     0.6793 |
| E5            |    1913389 |   1913389 | renda_gap          |                      6 |              58 |                 0.8708 |     0.6097 |              0.7179 |     0.5763 |     0.6627 |

### Árvore ternária Medicina — holdout

| experimento   |   n_treino |   n_teste | primeira_divisao   |   tree_depth_observada |   tree_n_leaves |   roc_auc_ovr_weighted |   accuracy |   balanced_accuracy |   f1_macro |   log_loss |
|:--------------|-----------:|----------:|:-------------------|-----------------------:|----------------:|-----------------------:|-----------:|--------------------:|-----------:|-----------:|
| E1            |     245203 |     61301 | gap                |                      6 |              57 |                 0.9730 |     0.8879 |              0.6727 |     0.5621 |     0.3231 |
| E2            |     245203 |     61301 | gap                |                      6 |              55 |                 0.9785 |     0.8955 |              0.7326 |     0.5923 |     0.2861 |
| E3            |     245203 |     61301 | gap                |                      6 |              57 |                 0.9797 |     0.8903 |              0.7335 |     0.5846 |     0.2833 |
| E4            |     245203 |     61301 | gap                |                      6 |              57 |                 0.9787 |     0.9019 |              0.7338 |     0.6024 |     0.2818 |
| E5            |     245203 |     61301 | gap                |                      6 |              55 |                 0.9792 |     0.8928 |              0.7376 |     0.5952 |     0.2775 |

### Árvore ternária Medicina — in-sample

| experimento   |   n_treino |   n_teste | primeira_divisao   |   tree_depth_observada |   tree_n_leaves |   roc_auc_ovr_weighted |   accuracy |   balanced_accuracy |   f1_macro |   log_loss |
|:--------------|-----------:|----------:|:-------------------|-----------------------:|----------------:|-----------------------:|-----------:|--------------------:|-----------:|-----------:|
| E1            |     306504 |    306504 | gap                |                      6 |              58 |                 0.9734 |     0.8877 |              0.6751 |     0.5561 |     0.3187 |
| E2            |     306504 |    306504 | gap                |                      6 |              56 |                 0.9782 |     0.8959 |              0.7323 |     0.5930 |     0.2843 |
| E3            |     306504 |    306504 | gap                |                      6 |              58 |                 0.9789 |     0.8929 |              0.7348 |     0.5888 |     0.2830 |
| E4            |     306504 |    306504 | gap                |                      6 |              57 |                 0.9789 |     0.8971 |              0.7347 |     0.5973 |     0.2793 |
| E5            |     306504 |    306504 | gap                |                      6 |              53 |                 0.9820 |     0.9030 |              0.7318 |     0.5848 |     0.2582 |

## Matrizes de confusão e métricas por classe — holdout E5

### Binária geral — holdout E5

| Classe real    |   Não contratado |   Contratada |
|:---------------|-----------------:|-------------:|
| Não contratado |           134817 |        69982 |
| Contratada     |             4311 |        27745 |

| classe         |   support |   precisao |   recall |     f1 |
|:---------------|----------:|-----------:|---------:|-------:|
| Não contratado |    204799 |     0.9690 |   0.6583 | 0.7840 |
| Contratada     |     32056 |     0.2839 |   0.8655 | 0.4276 |

### Ternária geral — holdout E5

| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |            112457 |             9663 |        23703 |
| Não contratado  |             24849 |            90836 |        89114 |
| Contratada      |              1189 |              798 |        30069 |

| classe          |   support |   precisao |   recall |     f1 |
|:----------------|----------:|-----------:|---------:|-------:|
| Lista de espera |    145823 |     0.8120 |   0.7712 | 0.7911 |
| Não contratado  |    204799 |     0.8967 |   0.4435 | 0.5935 |
| Contratada      |     32056 |     0.2104 |   0.9380 | 0.3438 |

### Ternária Medicina — holdout E5

| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |             51822 |             3615 |         1134 |
| Não contratado  |                70 |             1829 |         1372 |
| Contratada      |                34 |              349 |         1076 |

| classe          |   support |   precisao |   recall |     f1 |
|:----------------|----------:|-----------:|---------:|-------:|
| Lista de espera |     56571 |     0.9980 |   0.9161 | 0.9553 |
| Não contratado  |      3271 |     0.3157 |   0.5592 | 0.4036 |
| Contratada      |      1459 |     0.3004 |   0.7375 | 0.4269 |

## Importância agregada das variáveis — top 10, holdout E5

### Binária geral — holdout E5

| variavel_original         | bloco_label                   |   importancia_normalizada |
|:--------------------------|:------------------------------|--------------------------:|
| opcao_curso               | Opção de curso                |                    0.8499 |
| renda_per_capita          | Renda familiar per capita     |                    0.0554 |
| subarea_conhecimento      | Subárea de conhecimento       |                    0.0303 |
| renda_gap                 | Interação renda × desempenho  |                    0.0189 |
| turno                     | Turno                         |                    0.0189 |
| idade                     | Idade                         |                    0.0068 |
| nome_cine_area_geral      | Área geral CINE               |                    0.0065 |
| semestre                  | Semestre do processo seletivo |                    0.0060 |
| regiao_ies_alvo           | Região da oferta              |                    0.0030 |
| nome_cine_area_especifica | Área específica CINE          |                    0.0014 |

### Ternária geral — holdout E5

| variavel_original       | bloco_label                         |   importancia_normalizada |
|:------------------------|:------------------------------------|--------------------------:|
| renda_gap               | Interação renda × desempenho        |                    0.2994 |
| opcao_curso             | Opção de curso                      |                    0.2750 |
| nota_corte_gp           | Nota de corte do grupo              |                    0.1749 |
| ano                     | Ano do processo seletivo            |                    0.0736 |
| conceito_curso_gp       | Conceito do curso                   |                    0.0667 |
| subarea_conhecimento    | Subárea de conhecimento             |                    0.0360 |
| gap                     | Desempenho relativo à nota de corte |                    0.0330 |
| concluiu_curso_superior | Concluiu curso superior             |                    0.0159 |
| semestre                | Semestre do processo seletivo       |                    0.0120 |
| renda_per_capita        | Renda familiar per capita           |                    0.0100 |

### Ternária Medicina — holdout E5

| variavel_original             | bloco_label                         |   importancia_normalizada |
|:------------------------------|:------------------------------------|--------------------------:|
| gap                           | Desempenho relativo à nota de corte |                    0.8039 |
| opcao_curso                   | Opção de curso                      |                    0.0958 |
| nota_corte_gp                 | Nota de corte do grupo              |                    0.0233 |
| ano                           | Ano do processo seletivo            |                    0.0193 |
| uf_local_oferta               | UF do local de oferta               |                    0.0155 |
| concluiu_curso_superior       | Concluiu curso superior             |                    0.0089 |
| natureza_juridica_mantenedora | Natureza jurídica da mantenedora    |                    0.0069 |
| renda_gap                     | Interação renda × desempenho        |                    0.0068 |
| conceito_curso_gp             | Conceito do curso                   |                    0.0060 |
| regiao_ies_alvo               | Região da oferta                    |                    0.0050 |

## Probabilidade prevista de contratação — médias no holdout E5

### Binária geral — holdout E5

|   target_binario |     count |   mean |   median |   min |   max |
|-----------------:|----------:|-------:|---------:|------:|------:|
|             0.00 | 204799.00 |  31.45 |     9.48 |  0.59 | 86.18 |
|             1.00 |  32056.00 |  68.37 |    68.69 |  0.59 | 86.18 |

### Ternária geral — holdout E5

|   target_ternario |     count |   mean |   median |   min |   max |
|------------------:|----------:|-------:|---------:|------:|------:|
|              0.00 | 145823.00 |  10.28 |     1.27 |  0.00 | 77.94 |
|              1.00 | 204799.00 |  28.13 |     8.15 |  0.00 | 77.94 |
|              2.00 |  32056.00 |  61.45 |    67.40 |  0.00 | 77.94 |

### Ternária Medicina — holdout E5

|   target_ternario |    count |   mean |   median |   min |   max |
|------------------:|---------:|-------:|---------:|------:|------:|
|              0.00 | 56571.00 |   4.68 |     0.76 |  0.00 | 78.46 |
|              1.00 |  3271.00 |  40.96 |    44.58 |  0.00 | 64.50 |
|              2.00 |  1459.00 |  54.78 |    63.01 |  0.00 | 78.46 |

## Leitura visual das figuras

- As figuras de árvore existem em PDF e PNG, com dimensão alta; a árvore está legível como artefato visual, mas naturalmente densa por usar profundidade 6.

- Nos gráficos de importância, a binária geral é dominada por `opcao_curso`; a ternária geral por `renda_gap`, `opcao_curso` e `nota_corte_gp`; Medicina por `gap`.

- Os heatmaps de probabilidade prevista mostram padrão compatível com os dados: em Medicina a probabilidade de contratação é concentrada no intervalo `0 a +50`; no ternário geral a contratação sobe com desempenho; no binário geral a árvore produz superfícies em degraus, esperado para árvore de decisão.

## Veredito

Os resultados de árvore estão tecnicamente consistentes, com CSVs carregáveis, métricas coerentes com as matrizes de confusão, divisão holdout com tamanhos corretos e figuras/tabelas geradas nas pastas esperadas. Como modelo interpretável, a árvore é útil sobretudo para descrever regras e importâncias; para classificação da classe `Contratada`, a precisão continua baixa nos recortes geral e Medicina, então a interpretação deve enfatizar associação/estrutura de decisão, não predição individual perfeita.
