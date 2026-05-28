# Análise das árvores de decisão com profundidades 10, 14 e 19

Arquivo analisado: `tree_profundidades_para_analise.zip`.

## 1. Checagem estrutural

- Total de itens no ZIP: **1168**.
- CSVs: **206**; PDFs: **162**; PNGs: **162**; JSONs: **150**; logs: **39**.
- CSVs lidos com sucesso: **206**.
- CSVs com erro de leitura: **0**.
- Métricas de experimentos encontradas: **18** arquivos.
- Matrizes de confusão E5 encontradas: **18** arquivos.
- Figuras PNG de árvore/probabilidade/importância/matriz: **162** arquivos.
- Logs de artigo/apêndice das profundidades 10, 14 e 19: sem `Traceback` ou erro textual detectado.

**Observação de consistência:** esses resultados foram gerados antes da correção posterior que removeu `nome_cine_area_geral` e `nome_cine_area_especifica` da modelagem. Portanto, em alguns CSVs/figuras de importância ainda aparecem `Área geral CINE` e `Área específica CINE`. Para leitura estrita do artigo final, trate esses resultados como versão anterior; depois da correção, a modelagem precisa ser reexecutada para ficar apenas com `subarea_conhecimento` nesse bloco.

## 2. E5 holdout: comparação direta das profundidades

| modelo            |   profundidade |   n_teste |   ROC-AUC |   acc_bal |     F1 |   log_loss |   folhas | primeira_divisão   |   min_leaf |
|:------------------|---------------:|----------:|----------:|----------:|-------:|-----------:|---------:|:-------------------|-----------:|
| binário geral     |             10 |    236855 |    0.8445 |    0.7727 | 0.4363 |     0.4895 |      324 | opcao_curso        |        474 |
| binário geral     |             14 |    236855 |    0.8485 |    0.7748 | 0.4428 |     0.4865 |      759 | opcao_curso        |        474 |
| binário geral     |             19 |    236855 |    0.849  |    0.7751 | 0.442  |     0.4918 |     1145 | opcao_curso        |        474 |
| ternário geral    |             10 |    382678 |    0.91   |    0.7552 | 0.6546 |     0.5834 |      380 | renda_gap          |        765 |
| ternário geral    |             14 |    382678 |    0.9205 |    0.7645 | 0.665  |     0.5556 |      877 | renda_gap          |        765 |
| ternário geral    |             19 |    382678 |    0.9227 |    0.768  | 0.6633 |     0.551  |     1244 | renda_gap          |        765 |
| ternário medicina |             10 |     61301 |    0.985  |    0.7315 | 0.5933 |     0.2469 |      136 | gap                |        400 |
| ternário medicina |             14 |     61301 |    0.9853 |    0.7269 | 0.5953 |     0.2461 |      166 | gap                |        400 |
| ternário medicina |             19 |     61301 |    0.9852 |    0.7269 | 0.5953 |     0.2476 |      177 | gap                |        400 |

## 3. Melhor profundidade por métrica no E5 holdout

| modelo            |   melhor_ROC_AUC |   melhor_F1 |   melhor_log_loss | recomendação                                                           |
|:------------------|-----------------:|------------:|------------------:|:-----------------------------------------------------------------------|
| binário geral     |               19 |          14 |                14 | 14: melhor equilíbrio entre F1/log-loss; 19 só ganha pouco em ROC-AUC. |
| ternário geral    |               19 |          14 |                19 | 14 se priorizar F1; 19 se priorizar ROC-AUC/log-loss.                  |
| ternário medicina |               14 |          14 |                14 | 14: melhor ou empatado nas principais métricas.                        |

## 4. Estabilidade in-sample vs holdout no E5

| target   | recorte   |   depth |   roc_in |   roc_hold |   delta_roc |   balacc_in |   balacc_hold |   delta_balacc |    f1_in |   f1_hold |   delta_f1 |   logloss_in |   logloss_hold |   delta_logloss |   leaves_in |   leaves_hold |
|:---------|:----------|--------:|---------:|-----------:|------------:|------------:|--------------:|---------------:|---------:|----------:|-----------:|-------------:|---------------:|----------------:|------------:|--------------:|
| binario  | geral     |      10 | 0.846169 |   0.844517 |   -0.001652 |    0.772947 |      0.772676 |      -0.000271 | 0.43277  |  0.436266 |   0.003496 |     0.487896 |       0.489455 |        0.001559 |         329 |           324 |
| binario  | geral     |      14 | 0.852575 |   0.848511 |   -0.004064 |    0.778229 |      0.774806 |      -0.003423 | 0.444982 |  0.442819 |  -0.002163 |     0.480514 |       0.486472 |        0.005958 |         782 |           759 |
| binario  | geral     |      19 | 0.855239 |   0.849    |   -0.006239 |    0.780194 |      0.775107 |      -0.005087 | 0.445455 |  0.44203  |  -0.003426 |     0.477378 |       0.491821 |        0.014443 |        1190 |          1145 |
| ternario | geral     |      10 | 0.910403 |   0.910047 |   -0.000357 |    0.756218 |      0.755179 |      -0.001039 | 0.651891 |  0.654607 |   0.002716 |     0.581644 |       0.583401 |        0.001757 |         387 |           380 |
| ternario | geral     |      14 | 0.921006 |   0.920544 |   -0.000462 |    0.76651  |      0.764508 |      -0.002003 | 0.662418 |  0.664989 |   0.002571 |     0.551973 |       0.555558 |        0.003584 |         881 |           877 |
| ternario | geral     |      19 | 0.92338  |   0.922735 |   -0.000645 |    0.770268 |      0.768013 |      -0.002256 | 0.661093 |  0.663294 |   0.002201 |     0.54359  |       0.550969 |        0.007379 |        1259 |          1244 |
| ternario | medicina  |      10 | 0.986057 |   0.984967 |   -0.00109  |    0.746413 |      0.731543 |      -0.01487  | 0.608986 |  0.593341 |  -0.015646 |     0.234254 |       0.246934 |        0.01268  |         152 |           136 |
| ternario | medicina  |      14 | 0.9866   |   0.985295 |   -0.001305 |    0.749014 |      0.726887 |      -0.022127 | 0.607724 |  0.595306 |  -0.012418 |     0.231722 |       0.246095 |        0.014373 |         190 |           166 |
| ternario | medicina  |      19 | 0.986619 |   0.985224 |   -0.001395 |    0.749014 |      0.726887 |      -0.022127 | 0.607724 |  0.595306 |  -0.012418 |     0.231618 |       0.247572 |        0.015954 |         200 |           177 |

Interpretação: as diferenças entre in-sample e holdout são pequenas. A profundidade 19 aumenta a complexidade, mas o ganho de holdout é pequeno; no binário, o log loss piora em 19. Em Medicina, 14 e 19 ficam praticamente iguais na matriz de confusão, sugerindo saturação da árvore.

## 5. Evolução E1–E5 por modelo e profundidade

### binario | geral | profundidade 10
| experimento   |   roc_auc |   balanced_accuracy |     f1 |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|----------:|--------------------:|-------:|-----------:|----------------:|:-------------------|
| E1            |    0.6522 |              0.6071 | 0.2937 |     0.6547 |             430 | renda_gap          |
| E2            |    0.8281 |              0.7615 | 0.4192 |     0.5066 |             425 | opcao_curso        |
| E3            |    0.8379 |              0.7674 | 0.427  |     0.4973 |             390 | opcao_curso        |
| E4            |    0.8409 |              0.7692 | 0.4343 |     0.4938 |             388 | opcao_curso        |
| E5            |    0.8445 |              0.7727 | 0.4363 |     0.4895 |             324 | opcao_curso        |

### binario | geral | profundidade 14
| experimento   |   roc_auc |   balanced_accuracy |     f1 |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|----------:|--------------------:|-------:|-----------:|----------------:|:-------------------|
| E1            |    0.6566 |              0.6101 | 0.2961 |     0.651  |             984 | renda_gap          |
| E2            |    0.8286 |              0.7624 | 0.4207 |     0.5129 |            1075 | opcao_curso        |
| E3            |    0.8387 |              0.7678 | 0.4296 |     0.5007 |            1026 | opcao_curso        |
| E4            |    0.8425 |              0.7702 | 0.4352 |     0.4968 |            1034 | opcao_curso        |
| E5            |    0.8485 |              0.7748 | 0.4428 |     0.4865 |             759 | opcao_curso        |

### binario | geral | profundidade 19
| experimento   |   roc_auc |   balanced_accuracy |     f1 |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|----------:|--------------------:|-------:|-----------:|----------------:|:-------------------|
| E1            |    0.6615 |              0.6137 | 0.2987 |     0.6476 |            1407 | renda_gap          |
| E2            |    0.8286 |              0.7618 | 0.4214 |     0.5169 |            1356 | opcao_curso        |
| E3            |    0.8385 |              0.767  | 0.4313 |     0.5073 |            1351 | opcao_curso        |
| E4            |    0.8426 |              0.7705 | 0.4364 |     0.4998 |            1296 | opcao_curso        |
| E5            |    0.849  |              0.7751 | 0.442  |     0.4918 |            1145 | opcao_curso        |

### ternario | geral | profundidade 10
| experimento   |   roc_auc_ovr_weighted |   balanced_accuracy |   f1_macro |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|-----------------------:|--------------------:|-----------:|-----------:|----------------:|:-------------------|
| E1            |                 0.7631 |              0.5559 |     0.4611 |     0.898  |             556 | renda_gap          |
| E2            |                 0.8417 |              0.6853 |     0.5709 |     0.7093 |             454 | renda_gap          |
| E3            |                 0.8978 |              0.7429 |     0.6389 |     0.6128 |             445 | renda_gap          |
| E4            |                 0.9014 |              0.7473 |     0.6419 |     0.6067 |             440 | renda_gap          |
| E5            |                 0.91   |              0.7552 |     0.6546 |     0.5834 |             380 | renda_gap          |

### ternario | geral | profundidade 14
| experimento   |   roc_auc_ovr_weighted |   balanced_accuracy |   f1_macro |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|-----------------------:|--------------------:|-----------:|-----------:|----------------:|:-------------------|
| E1            |                 0.763  |              0.5586 |     0.455  |     0.8957 |            1181 | renda_gap          |
| E2            |                 0.8479 |              0.69   |     0.5745 |     0.7014 |            1082 | renda_gap          |
| E3            |                 0.9045 |              0.7495 |     0.6413 |     0.5987 |            1116 | renda_gap          |
| E4            |                 0.9103 |              0.7545 |     0.644  |     0.5848 |            1090 | renda_gap          |
| E5            |                 0.9205 |              0.7645 |     0.665  |     0.5556 |             877 | renda_gap          |

### ternario | geral | profundidade 19
| experimento   |   roc_auc_ovr_weighted |   balanced_accuracy |   f1_macro |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|-----------------------:|--------------------:|-----------:|-----------:|----------------:|:-------------------|
| E1            |                 0.7625 |              0.5618 |     0.4609 |     0.8943 |            1439 | renda_gap          |
| E2            |                 0.851  |              0.6914 |     0.5778 |     0.7002 |            1385 | renda_gap          |
| E3            |                 0.9054 |              0.7507 |     0.6419 |     0.5974 |            1380 | renda_gap          |
| E4            |                 0.9114 |              0.7562 |     0.6474 |     0.5833 |            1367 | renda_gap          |
| E5            |                 0.9227 |              0.768  |     0.6633 |     0.551  |            1244 | renda_gap          |

### ternario | medicina | profundidade 10
| experimento   |   roc_auc_ovr_weighted |   balanced_accuracy |   f1_macro |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|-----------------------:|--------------------:|-----------:|-----------:|----------------:|:-------------------|
| E1            |                 0.9735 |              0.6651 |     0.5548 |     0.322  |             176 | gap                |
| E2            |                 0.9806 |              0.7235 |     0.5753 |     0.2779 |             139 | gap                |
| E3            |                 0.9819 |              0.7276 |     0.5781 |     0.2691 |             160 | gap                |
| E4            |                 0.9819 |              0.736  |     0.5868 |     0.2634 |             155 | gap                |
| E5            |                 0.985  |              0.7315 |     0.5933 |     0.2469 |             136 | gap                |

### ternario | medicina | profundidade 14
| experimento   |   roc_auc_ovr_weighted |   balanced_accuracy |   f1_macro |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|-----------------------:|--------------------:|-----------:|-----------:|----------------:|:-------------------|
| E1            |                 0.9734 |              0.6637 |     0.5512 |     0.3264 |             230 | gap                |
| E2            |                 0.9806 |              0.7235 |     0.5753 |     0.2773 |             169 | gap                |
| E3            |                 0.982  |              0.7276 |     0.5781 |     0.2697 |             180 | gap                |
| E4            |                 0.9819 |              0.7336 |     0.585  |     0.2645 |             172 | gap                |
| E5            |                 0.9853 |              0.7269 |     0.5953 |     0.2461 |             166 | gap                |

### ternario | medicina | profundidade 19
| experimento   |   roc_auc_ovr_weighted |   balanced_accuracy |   f1_macro |   log_loss |   tree_n_leaves | primeira_divisao   |
|:--------------|-----------------------:|--------------------:|-----------:|-----------:|----------------:|:-------------------|
| E1            |                 0.9737 |              0.6637 |     0.5512 |     0.3261 |             272 | gap                |
| E2            |                 0.9805 |              0.7235 |     0.5753 |     0.2777 |             177 | gap                |
| E3            |                 0.982  |              0.7276 |     0.5781 |     0.2697 |             180 | gap                |
| E4            |                 0.9819 |              0.7336 |     0.585  |     0.265  |             180 | gap                |
| E5            |                 0.9852 |              0.7269 |     0.5953 |     0.2476 |             177 | gap                |

## 6. Matrizes de confusão E5 holdout e métricas por classe

### binário geral | profundidade 10
Matriz:
| Classe real    |   Não contratado |   Contratada |
|:---------------|-----------------:|-------------:|
| Não contratado |           135077 |        69722 |
| Contratada     |             3661 |        28395 |

Métricas por classe:
| classe         |   suporte |   precision |   recall |     f1 |
|:---------------|----------:|------------:|---------:|-------:|
| Não contratado |    204799 |      0.9736 |   0.6596 | 0.7864 |
| Contratada     |     32056 |      0.2894 |   0.8858 | 0.4363 |

### ternário geral | profundidade 10
Matriz:
| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |            124435 |            14113 |         7275 |
| Não contratado  |             20691 |           121200 |        62908 |
| Contratada      |              1804 |             3953 |        26299 |

Métricas por classe:
| classe          |   suporte |   precision |   recall |     f1 |
|:----------------|----------:|------------:|---------:|-------:|
| Lista de espera |    145823 |      0.8469 |   0.8533 | 0.8501 |
| Não contratado  |    204799 |      0.8703 |   0.5918 | 0.7045 |
| Contratada      |     32056 |      0.2726 |   0.8204 | 0.4092 |

### ternário Medicina | profundidade 10
Matriz:
| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |             52296 |             2777 |         1498 |
| Não contratado  |                80 |             1738 |         1453 |
| Contratada      |                35 |              346 |         1078 |

Métricas por classe:
| classe          |   suporte |   precision |   recall |     f1 |
|:----------------|----------:|------------:|---------:|-------:|
| Lista de espera |     56571 |      0.9978 |   0.9244 | 0.9597 |
| Não contratado  |      3271 |      0.3575 |   0.5313 | 0.4274 |
| Contratada      |      1459 |      0.2676 |   0.7389 | 0.3929 |

### binário geral | profundidade 14
Matriz:
| Classe real    |   Não contratado |   Contratada |
|:---------------|-----------------:|-------------:|
| Não contratado |           138294 |        66505 |
| Contratada     |             4028 |        28028 |

Métricas por classe:
| classe         |   suporte |   precision |   recall |     f1 |
|:---------------|----------:|------------:|---------:|-------:|
| Não contratado |    204799 |      0.9717 |   0.6753 | 0.7968 |
| Contratada     |     32056 |      0.2965 |   0.8743 | 0.4428 |

### ternário geral | profundidade 14
Matriz:
| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |            126318 |            12702 |         6803 |
| Não contratado  |             18583 |           123942 |        62274 |
| Contratada      |              1669 |             4034 |        26353 |

Métricas por classe:
| classe          |   suporte |   precision |   recall |     f1 |
|:----------------|----------:|------------:|---------:|-------:|
| Lista de espera |    145823 |      0.8618 |   0.8662 | 0.864  |
| Não contratado  |    204799 |      0.881  |   0.6052 | 0.7175 |
| Contratada      |     32056 |      0.2762 |   0.8221 | 0.4134 |

### ternário Medicina | profundidade 14
Matriz:
| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |             52296 |             2860 |         1415 |
| Não contratado  |                80 |             1847 |         1344 |
| Contratada      |                35 |              415 |         1009 |

Métricas por classe:
| classe          |   suporte |   precision |   recall |     f1 |
|:----------------|----------:|------------:|---------:|-------:|
| Lista de espera |     56571 |      0.9978 |   0.9244 | 0.9597 |
| Não contratado  |      3271 |      0.3606 |   0.5647 | 0.4401 |
| Contratada      |      1459 |      0.2678 |   0.6916 | 0.3861 |

### binário geral | profundidade 19
Matriz:
| Classe real    |   Não contratado |   Contratada |
|:---------------|-----------------:|-------------:|
| Não contratado |           137638 |        67161 |
| Contratada     |             3906 |        28150 |

Métricas por classe:
| classe         |   suporte |   precision |   recall |     f1 |
|:---------------|----------:|------------:|---------:|-------:|
| Não contratado |    204799 |      0.9724 |   0.6721 | 0.7948 |
| Contratada     |     32056 |      0.2953 |   0.8782 | 0.442  |

### ternário geral | profundidade 19
Matriz:
| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |            126860 |            11462 |         7501 |
| Não contratado  |             18654 |           121667 |        64478 |
| Contratada      |              1709 |             3420 |        26927 |

Métricas por classe:
| classe          |   suporte |   precision |   recall |     f1 |
|:----------------|----------:|------------:|---------:|-------:|
| Lista de espera |    145823 |      0.8617 |   0.87   | 0.8658 |
| Não contratado  |    204799 |      0.891  |   0.5941 | 0.7129 |
| Contratada      |     32056 |      0.2722 |   0.84   | 0.4112 |

### ternário Medicina | profundidade 19
Matriz:
| Classe real     |   Lista de espera |   Não contratado |   Contratada |
|:----------------|------------------:|-----------------:|-------------:|
| Lista de espera |             52296 |             2860 |         1415 |
| Não contratado  |                80 |             1847 |         1344 |
| Contratada      |                35 |              415 |         1009 |

Métricas por classe:
| classe          |   suporte |   precision |   recall |     f1 |
|:----------------|----------:|------------:|---------:|-------:|
| Lista de espera |     56571 |      0.9978 |   0.9244 | 0.9597 |
| Não contratado  |      3271 |      0.3606 |   0.5647 | 0.4401 |
| Contratada      |      1459 |      0.2678 |   0.6916 | 0.3861 |

## 7. Importâncias agregadas E5 holdout

### binario | geral | profundidade 10
| variavel_original         | bloco_label                         |   importancia_normalizada |
|:--------------------------|:------------------------------------|--------------------------:|
| opcao_curso               | Opção de curso                      |                    0.802  |
| renda_per_capita          | Renda familiar per capita           |                    0.0602 |
| subarea_conhecimento      | Subárea de conhecimento             |                    0.0356 |
| renda_gap                 | Interação renda × desempenho        |                    0.0209 |
| turno                     | Turno                               |                    0.019  |
| semestre                  | Semestre do processo seletivo       |                    0.0111 |
| nome_cine_area_geral      | Área geral CINE                     |                    0.0091 |
| idade                     | Idade                               |                    0.0086 |
| conceito_curso_gp         | Conceito do curso                   |                    0.0086 |
| nota_corte_gp             | Nota de corte do grupo              |                    0.0061 |
| ano                       | Ano do processo seletivo            |                    0.0043 |
| nome_cine_area_especifica | Área específica CINE                |                    0.0042 |
| regiao_ies_alvo           | Região da oferta                    |                    0.0031 |
| gap                       | Desempenho relativo à nota de corte |                    0.0021 |
| uf_local_oferta           | UF do local de oferta               |                    0.0014 |

### ternario | geral | profundidade 10
| variavel_original       | bloco_label                         |   importancia_normalizada |
|:------------------------|:------------------------------------|--------------------------:|
| renda_gap               | Interação renda × desempenho        |                    0.2605 |
| opcao_curso             | Opção de curso                      |                    0.2385 |
| nota_corte_gp           | Nota de corte do grupo              |                    0.1682 |
| ano                     | Ano do processo seletivo            |                    0.0769 |
| conceito_curso_gp       | Conceito do curso                   |                    0.0695 |
| gap                     | Desempenho relativo à nota de corte |                    0.0631 |
| subarea_conhecimento    | Subárea de conhecimento             |                    0.0366 |
| semestre                | Semestre do processo seletivo       |                    0.0246 |
| concluiu_curso_superior | Concluiu curso superior             |                    0.016  |
| uf_local_oferta         | UF do local de oferta               |                    0.0128 |
| renda_per_capita        | Renda familiar per capita           |                    0.0122 |
| regiao_ies_alvo         | Região da oferta                    |                    0.0076 |
| regiao_morar            | Região de residência                |                    0.0049 |
| idade                   | Idade                               |                    0.0028 |
| nome_cine_area_geral    | Área geral CINE                     |                    0.0024 |

### ternario | medicina | profundidade 10
| variavel_original             | bloco_label                         |   importancia_normalizada |
|:------------------------------|:------------------------------------|--------------------------:|
| gap                           | Desempenho relativo à nota de corte |                    0.7796 |
| opcao_curso                   | Opção de curso                      |                    0.0913 |
| nota_corte_gp                 | Nota de corte do grupo              |                    0.0354 |
| ano                           | Ano do processo seletivo            |                    0.0215 |
| uf_local_oferta               | UF do local de oferta               |                    0.0153 |
| concluiu_curso_superior       | Concluiu curso superior             |                    0.0141 |
| natureza_juridica_mantenedora | Natureza jurídica da mantenedora    |                    0.01   |
| organizacao_academica         | Organização acadêmica               |                    0.0086 |
| renda_gap                     | Interação renda × desempenho        |                    0.0072 |
| regiao_ies_alvo               | Região da oferta                    |                    0.0044 |
| semestre                      | Semestre do processo seletivo       |                    0.0041 |
| conceito_curso_gp             | Conceito do curso                   |                    0.0039 |
| renda_per_capita              | Renda familiar per capita           |                    0.0036 |
| regiao_morar                  | Região de residência                |                    0.0009 |
| idade                         | Idade                               |                    0.0001 |

### binario | geral | profundidade 14
| variavel_original             | bloco_label                      |   importancia_normalizada |
|:------------------------------|:---------------------------------|--------------------------:|
| opcao_curso                   | Opção de curso                   |                    0.7795 |
| renda_per_capita              | Renda familiar per capita        |                    0.0639 |
| subarea_conhecimento          | Subárea de conhecimento          |                    0.0364 |
| renda_gap                     | Interação renda × desempenho     |                    0.0218 |
| turno                         | Turno                            |                    0.0187 |
| semestre                      | Semestre do processo seletivo    |                    0.0111 |
| nome_cine_area_geral          | Área geral CINE                  |                    0.0102 |
| idade                         | Idade                            |                    0.0097 |
| conceito_curso_gp             | Conceito do curso                |                    0.0092 |
| nota_corte_gp                 | Nota de corte do grupo           |                    0.008  |
| nome_cine_area_especifica     | Área específica CINE             |                    0.0066 |
| ano                           | Ano do processo seletivo         |                    0.0055 |
| regiao_ies_alvo               | Região da oferta                 |                    0.0041 |
| natureza_juridica_mantenedora | Natureza jurídica da mantenedora |                    0.0033 |
| organizacao_academica         | Organização acadêmica            |                    0.0032 |

### ternario | geral | profundidade 14
| variavel_original       | bloco_label                         |   importancia_normalizada |
|:------------------------|:------------------------------------|--------------------------:|
| renda_gap               | Interação renda × desempenho        |                    0.2511 |
| opcao_curso             | Opção de curso                      |                    0.2282 |
| nota_corte_gp           | Nota de corte do grupo              |                    0.1697 |
| ano                     | Ano do processo seletivo            |                    0.0756 |
| conceito_curso_gp       | Conceito do curso                   |                    0.0671 |
| gap                     | Desempenho relativo à nota de corte |                    0.0649 |
| subarea_conhecimento    | Subárea de conhecimento             |                    0.038  |
| semestre                | Semestre do processo seletivo       |                    0.0251 |
| uf_local_oferta         | UF do local de oferta               |                    0.0179 |
| concluiu_curso_superior | Concluiu curso superior             |                    0.0178 |
| renda_per_capita        | Renda familiar per capita           |                    0.0143 |
| regiao_ies_alvo         | Região da oferta                    |                    0.0093 |
| regiao_morar            | Região de residência                |                    0.0075 |
| nome_cine_area_geral    | Área geral CINE                     |                    0.0049 |
| idade                   | Idade                               |                    0.003  |

### ternario | medicina | profundidade 14
| variavel_original             | bloco_label                         |   importancia_normalizada |
|:------------------------------|:------------------------------------|--------------------------:|
| gap                           | Desempenho relativo à nota de corte |                    0.7746 |
| opcao_curso                   | Opção de curso                      |                    0.0906 |
| nota_corte_gp                 | Nota de corte do grupo              |                    0.0377 |
| ano                           | Ano do processo seletivo            |                    0.0214 |
| uf_local_oferta               | UF do local de oferta               |                    0.0152 |
| concluiu_curso_superior       | Concluiu curso superior             |                    0.014  |
| natureza_juridica_mantenedora | Natureza jurídica da mantenedora    |                    0.0113 |
| organizacao_academica         | Organização acadêmica               |                    0.0099 |
| renda_gap                     | Interação renda × desempenho        |                    0.0072 |
| renda_per_capita              | Renda familiar per capita           |                    0.0047 |
| regiao_ies_alvo               | Região da oferta                    |                    0.0045 |
| semestre                      | Semestre do processo seletivo       |                    0.004  |
| conceito_curso_gp             | Conceito do curso                   |                    0.0039 |
| regiao_morar                  | Região de residência                |                    0.0009 |
| idade                         | Idade                               |                    0.0001 |

### binario | geral | profundidade 19
| variavel_original         | bloco_label                         |   importancia_normalizada |
|:--------------------------|:------------------------------------|--------------------------:|
| opcao_curso               | Opção de curso                      |                    0.7698 |
| renda_per_capita          | Renda familiar per capita           |                    0.0652 |
| subarea_conhecimento      | Subárea de conhecimento             |                    0.0362 |
| renda_gap                 | Interação renda × desempenho        |                    0.0226 |
| turno                     | Turno                               |                    0.0186 |
| nome_cine_area_geral      | Área geral CINE                     |                    0.0112 |
| semestre                  | Semestre do processo seletivo       |                    0.011  |
| idade                     | Idade                               |                    0.0103 |
| nota_corte_gp             | Nota de corte do grupo              |                    0.0096 |
| conceito_curso_gp         | Conceito do curso                   |                    0.0096 |
| nome_cine_area_especifica | Área específica CINE                |                    0.0069 |
| ano                       | Ano do processo seletivo            |                    0.0059 |
| gap                       | Desempenho relativo à nota de corte |                    0.0046 |
| regiao_ies_alvo           | Região da oferta                    |                    0.0046 |
| organizacao_academica     | Organização acadêmica               |                    0.0038 |

### ternario | geral | profundidade 19
| variavel_original         | bloco_label                         |   importancia_normalizada |
|:--------------------------|:------------------------------------|--------------------------:|
| renda_gap                 | Interação renda × desempenho        |                    0.249  |
| opcao_curso               | Opção de curso                      |                    0.2257 |
| nota_corte_gp             | Nota de corte do grupo              |                    0.1704 |
| ano                       | Ano do processo seletivo            |                    0.0756 |
| conceito_curso_gp         | Conceito do curso                   |                    0.0665 |
| gap                       | Desempenho relativo à nota de corte |                    0.0648 |
| subarea_conhecimento      | Subárea de conhecimento             |                    0.0382 |
| semestre                  | Semestre do processo seletivo       |                    0.0251 |
| uf_local_oferta           | UF do local de oferta               |                    0.0184 |
| concluiu_curso_superior   | Concluiu curso superior             |                    0.0176 |
| renda_per_capita          | Renda familiar per capita           |                    0.015  |
| regiao_ies_alvo           | Região da oferta                    |                    0.0097 |
| regiao_morar              | Região de residência                |                    0.0077 |
| nome_cine_area_geral      | Área geral CINE                     |                    0.0053 |
| nome_cine_area_especifica | Área específica CINE                |                    0.0036 |

### ternario | medicina | profundidade 19
| variavel_original             | bloco_label                         |   importancia_normalizada |
|:------------------------------|:------------------------------------|--------------------------:|
| gap                           | Desempenho relativo à nota de corte |                    0.7746 |
| opcao_curso                   | Opção de curso                      |                    0.0906 |
| nota_corte_gp                 | Nota de corte do grupo              |                    0.0377 |
| ano                           | Ano do processo seletivo            |                    0.0214 |
| uf_local_oferta               | UF do local de oferta               |                    0.0152 |
| concluiu_curso_superior       | Concluiu curso superior             |                    0.014  |
| natureza_juridica_mantenedora | Natureza jurídica da mantenedora    |                    0.0113 |
| organizacao_academica         | Organização acadêmica               |                    0.0099 |
| renda_gap                     | Interação renda × desempenho        |                    0.0072 |
| renda_per_capita              | Renda familiar per capita           |                    0.0047 |
| regiao_ies_alvo               | Região da oferta                    |                    0.0045 |
| semestre                      | Semestre do processo seletivo       |                    0.004  |
| conceito_curso_gp             | Conceito do curso                   |                    0.0039 |
| regiao_morar                  | Região de residência                |                    0.0009 |
| idade                         | Idade                               |                    0.0001 |

## 8. Probabilidades previstas por classe real

|   depth | target   | recorte   | avaliacao     |   classe_real |      n |   prob_contratacao_mean |   prob_contratacao_median |   prob_contratacao_min |   prob_contratacao_max |
|--------:|:---------|:----------|:--------------|--------------:|-------:|------------------------:|--------------------------:|-----------------------:|-----------------------:|
|      10 | binario  | geral     | holdout_80_20 |             0 | 204799 |                 30.3622 |                   15.6704 |                      0 |                94.7864 |
|      10 | binario  | geral     | holdout_80_20 |             1 |  32056 |                 69.3913 |                   73.7157 |                      0 |                94.7864 |
|      14 | binario  | geral     | holdout_80_20 |             0 | 204799 |                 29.8229 |                   15.9764 |                      0 |                94.7864 |
|      14 | binario  | geral     | holdout_80_20 |             1 |  32056 |                 69.6962 |                   72.9756 |                      0 |                94.7864 |
|      19 | binario  | geral     | holdout_80_20 |             0 | 204799 |                 29.6006 |                   16.1175 |                      0 |                94.7864 |
|      19 | binario  | geral     | holdout_80_20 |             1 |  32056 |                 69.7284 |                   74.2818 |                      0 |                94.7864 |
|      10 | ternario | geral     | holdout_80_20 |             0 | 145823 |                  7.8765 |                    0.6408 |                      0 |                85.9509 |
|      10 | ternario | geral     | holdout_80_20 |             1 | 204799 |                 27.971  |                    9.4172 |                      0 |                89.0362 |
|      10 | ternario | geral     | holdout_80_20 |             2 |  32056 |                 63.9652 |                   68.0417 |                      0 |                89.0362 |
|      14 | ternario | geral     | holdout_80_20 |             0 | 145823 |                  7.2031 |                    0.3569 |                      0 |                93.1432 |
|      14 | ternario | geral     | holdout_80_20 |             1 | 204799 |                 27.7206 |                   10.8418 |                      0 |                93.1432 |
|      14 | ternario | geral     | holdout_80_20 |             2 |  32056 |                 64.7129 |                   70.2955 |                      0 |                93.1432 |
|      19 | ternario | geral     | holdout_80_20 |             0 | 145823 |                  7.0804 |                    0.3932 |                      0 |                93.1432 |
|      19 | ternario | geral     | holdout_80_20 |             1 | 204799 |                 27.5167 |                   10.9953 |                      0 |                93.1432 |
|      19 | ternario | geral     | holdout_80_20 |             2 |  32056 |                 64.9483 |                   69.632  |                      0 |                93.1432 |
|      10 | ternario | medicina  | holdout_80_20 |             0 |  56571 |                  3.9312 |                    0.227  |                      0 |                76.4364 |
|      10 | ternario | medicina  | holdout_80_20 |             1 |   3271 |                 40.2824 |                   42.3194 |                      0 |                76.4364 |
|      10 | ternario | medicina  | holdout_80_20 |             2 |   1459 |                 55.1349 |                   59.9042 |                      0 |                76.4364 |
|      14 | ternario | medicina  | holdout_80_20 |             0 |  56571 |                  3.8927 |                    0      |                      0 |                77.3017 |
|      14 | ternario | medicina  | holdout_80_20 |             1 |   3271 |                 40.1468 |                   42.3828 |                      0 |                77.3017 |
|      14 | ternario | medicina  | holdout_80_20 |             2 |   1459 |                 55.109  |                   61.7492 |                      0 |                77.3017 |
|      19 | ternario | medicina  | holdout_80_20 |             0 |  56571 |                  3.8935 |                    0      |                      0 |                77.3017 |
|      19 | ternario | medicina  | holdout_80_20 |             1 |   3271 |                 40.1468 |                   42.3828 |                      0 |                77.3017 |
|      19 | ternario | medicina  | holdout_80_20 |             2 |   1459 |                 55.1083 |                   61.7492 |                      0 |                77.3017 |

## 9. Leitura substantiva


### Binário geral
As três profundidades mantêm o mesmo padrão: primeira divisão por `opcao_curso`, alta importância dessa variável e boa sensibilidade para a classe `Contratada`. A precisão da classe `Contratada` continua baixa, porque o modelo prevê muitos falsos positivos. A profundidade 14 é a melhor escolha pragmática: tem o melhor F1 e o menor log loss; a profundidade 19 só aumenta bastante o número de folhas e melhora muito pouco o ROC-AUC.

### Ternário geral
A primeira divisão fica em `renda_gap` nas três profundidades. Isso reforça a leitura de que, quando a situação é decomposta em lista de espera, não contratado e contratada, a interação entre condição econômica e desempenho acadêmico organiza fortemente a separação dos grupos. A profundidade 19 tem o maior ROC-AUC e menor log loss; a profundidade 14 tem o maior F1 macro. A diferença é pequena, então 14 é mais enxuta e 19 é mais agressiva.

### Ternário Medicina
A primeira divisão é `gap` em todas as profundidades. O desempenho acadêmico domina as importâncias, com cerca de 0,77–0,78 no E5. A profundidade 14 é suficiente: a profundidade 19 praticamente repete a matriz de confusão da 14, com mais folhas e sem ganho material. A classe `Lista de espera` é muito bem separada, enquanto `Contratada` e `Não contratado` continuam difíceis por desbalanceamento e proximidade substantiva.

### Sobre as figuras
As figuras de probabilidade prevista são coerentes com árvores: mostram superfícies em blocos, não curvas suaves. No binário geral, a probabilidade tende a ser maior nas faixas de melhor desempenho e menor renda, com alguma não monotonicidade por causa das partições. No ternário geral, a estrutura é mais ordenada: desempenho maior aumenta a probabilidade prevista, e renda maior tende a reduzir em boa parte da matriz. Em Medicina, o padrão é fortemente concentrado no desempenho, especialmente na faixa `0 a +50`, o que deve ser lido como partição do modelo, não como relação causal monotônica.

## 10. Veredito


- Execução: OK.
- CSVs: OK.
- Logs: OK.
- Matrizes: OK.
- Figuras: OK.
- Holdout usando tamanho correto: OK.
- As árvores profundas não mostram overfitting grave, mas profundidade 19 aumenta complexidade com ganhos pequenos.
- Para relatório/artigo, a profundidade 14 é a melhor escolha geral de compromisso.
- Para comparação metodológica, manter 10/14/19 no apêndice é útil.
- Para a versão final após correção das variáveis CINE, os modelos devem ser reexecutados para remover `nome_cine_area_geral` e `nome_cine_area_especifica` da modelagem.
