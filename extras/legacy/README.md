# FIES 2019–2021: pipeline KDD para análise da conversão em contrato

Este repositório contém o pipeline de dados e os scripts utilizados no artigo **"Elegibilidade acadêmica e contratação no FIES: fatores associados à conversão em contrato nos grupos de baixa renda, 2019–2021"**.

O objetivo é permitir a reprodução das etapas de preparação dos dados, construção das bases analíticas, geração de tabelas e figuras, ajuste dos modelos estatísticos e produção dos apêndices do artigo.

---

## 1. Visão geral

O projeto segue uma organização inspirada no processo KDD (*Knowledge Discovery in Databases*). No contexto deste repositório, o KDD inclui a preparação dos dados, a construção de variáveis, a criação de bases analíticas, a modelagem e a interpretação dos resultados.

A etapa de ETL é usada para transformar os microdados brutos em bases limpas, padronizadas e auditáveis. A etapa de modelagem é executada depois, a partir de ABTs específicas.

```text
Dados brutos
   ↓
Staging
   ↓
Transformação
   ↓
Base curada
   ↓
ABTs
   ↓
Modelos salvos
   ↓
Tabelas, figuras e apêndices do artigo
```

Neste projeto:

- **Staging**: padronização inicial dos arquivos brutos.
- **Transformação**: unificação, enriquecimento, correções e classificação das modalidades.
- **Base curada**: base final limpa e pronta para análise.
- **ABT**: base analítica específica para modelagem.
- **Modelos salvos**: modelos previamente ajustados e armazenados para reprodução rápida das tabelas e figuras.
- **Reports**: figuras, tabelas e apêndices usados no artigo.

---

## 2. Dados utilizados

O projeto utiliza microdados públicos de duas fontes.

### 2.1 FIES

Microdados de inscrições e ofertas do Fundo de Financiamento Estudantil, referentes ao período de 2019 a 2021.

Fonte: Portal de Dados Abertos do MEC  
https://dadosabertos.mec.gov.br/fies

### 2.2 Censo da Educação Superior

Microdados do Censo da Educação Superior, utilizados para enriquecer os registros do FIES com informações de curso e classificação CINE.

Fonte: INEP  
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior

Os arquivos brutos não são versionados neste repositório em razão do tamanho dos microdados. Para reproduzir a análise, baixe os arquivos originais e coloque-os nas pastas indicadas na seção 5.

---

## 3. Estrutura recomendada do repositório

A estrutura separa dados, código, modelos e relatórios. Essa separação evita misturar bases intermediárias com figuras, tabelas e arquivos de modelagem.

```text
federal-microdata-kdd/
│
├── data/
│   ├── 01_raw/
│   │   ├── fies/
│   │   └── inep/
│   │
│   ├── 02_staging/
│   │   ├── fies/
│   │   └── inep/
│   │
│   ├── 03_interim/
│   │   ├── fies/
│   │   ├── inep/
│   │   └── temporarios/
│   │
│   ├── 04_curated/
│   │   ├── parquet/
│   │   └── sqlite/
│   │
│   └── 05_abt/
│       ├── abt_contratacao_binaria_geral.parquet
│       ├── abt_contratacao_ternaria_geral.parquet
│       ├── abt_contratacao_binaria_medicina.parquet
│       └── abt_contratacao_ternaria_medicina.parquet
│
├── models/
│   ├── general/
│   │   ├── logit_binario.pkl
│   │   ├── arvore_binaria.pkl
│   │   ├── logit_multinomial.pkl
│   │   └── arvore_ternaria.pkl
│   │
│   └── medicina/
│       ├── logit_binario.pkl
│       ├── arvore_binaria.pkl
│       ├── logit_multinomial.pkl
│       └── arvore_ternaria.pkl
│
├── reports/
│   ├── article/
│   │   ├── figures/
│   │   ├── tables/
│   │   └── appendix/
│   │
│   ├── diagnostics/
│   └── logs/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   │
│   ├── pipeline/
│   │   ├── staging.py
│   │   ├── transform_01_unificacao.py
│   │   ├── transform_02_cine.py
│   │   ├── transform_03_modalidade.py
│   │   └── curate.py
│   │
│   ├── abt/
│   │   ├── build_abt_binaria.py
│   │   └── build_abt_ternaria.py
│   │
│   ├── modeling/
│   │   ├── fit_logit_binario.py
│   │   ├── fit_arvore_binaria.py
│   │   ├── fit_logit_multinomial.py
│   │   ├── fit_arvore_ternaria.py
│   │   └── evaluate.py
│   │
│   ├── article/
│   │   ├── 01_fluxo_selecao.py
│   │   ├── 02_matrizes_renda_desempenho.py
│   │   ├── 03_financiamento_coparticipacao.py
│   │   ├── 04_modelos_gerais.py
│   │   ├── 05_medicina.py
│   │   └── 06_apendices.py
│   │
│   └── utils/
│       ├── io.py
│       ├── validation.py
│       └── plotting.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Preparação do ambiente

Na raiz do projeto, crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Verifique se o ambiente foi configurado corretamente:

```bash
python3 main.py check
```

---

## 5. Organização dos dados brutos

Após baixar os arquivos originais, organize os dados na pasta `data/01_raw/`.

Estrutura esperada:

```text
data/01_raw/
├── fies/
│   ├── inscricoes_2019_1.csv
│   ├── inscricoes_2019_2.csv
│   ├── inscricoes_2020_1.csv
│   ├── inscricoes_2020_2.csv
│   ├── inscricoes_2021_1.csv
│   ├── inscricoes_2021_2.csv
│   ├── ofertas_2019_1.csv
│   ├── ofertas_2019_2.csv
│   ├── ofertas_2020_1.csv
│   ├── ofertas_2020_2.csv
│   ├── ofertas_2021_1.csv
│   └── ofertas_2021_2.csv
│
└── inep/
    ├── 2016/
    ├── 2017/
    ├── 2018/
    ├── 2019/
    ├── 2020/
    ├── 2021/
    ├── 2022/
    ├── 2023/
    └── 2024/
```

Os nomes dos arquivos podem variar conforme o padrão disponibilizado pelos órgãos oficiais. O script de staging deve localizar, padronizar e renomear os arquivos internamente.

---

## 6. Reprodução rápida

Para executar todo o pipeline, gerar as ABTs, usar os modelos já salvos e produzir as figuras e tabelas do artigo:

```bash
python3 main.py reproduce all --use-saved-models
```

Esse comando é recomendado para reprodução rápida, porque evita reajustar modelos que já foram salvos.

Para refazer os modelos do zero, use:

```bash
python3 main.py reproduce all --refit-models
```

A etapa de ajuste dos modelos pode levar tempo, dependendo do computador, da quantidade de registros e do número de variáveis incluídas. Por isso, o repositório permite separar a reprodução das saídas do artigo do reajuste completo dos modelos.

---

## 7. Execução por etapas

### 7.1 Staging

Padroniza os arquivos brutos, remove duplicatas exatas e organiza os arquivos em uma estrutura comum.

```bash
python3 main.py pipeline staging
```

Saídas esperadas:

```text
data/02_staging/fies/
data/02_staging/inep/
reports/logs/staging.log
```

### 7.2 Transformação

Unifica os arquivos por ano e semestre, realiza os cruzamentos com o Censo da Educação Superior, aplica correções de classificação CINE e classifica os registros conforme as regras de modalidade do FIES.

```bash
python3 main.py pipeline transform
```

Saídas esperadas:

```text
data/03_interim/fies/
data/03_interim/inep/
reports/logs/transform.log
```

### 7.3 Curadoria da base final

Gera a base final limpa, enriquecida, auditada e pronta para análise em formato Parquet e SQLite.

```bash
python3 main.py pipeline curate
```

Saídas esperadas:

```text
data/04_curated/parquet/
data/04_curated/sqlite/
reports/logs/curate.log
```

### 7.4 Pipeline completo de dados

Executa as etapas de staging, transformação e curadoria.

```bash
python3 main.py pipeline all
```

---

## 8. Construção das ABTs

As ABTs são criadas a partir da base curada, mas possuem filtros, variáveis e codificações específicos para cada modelo.

```bash
python3 main.py abt all
```

Saídas esperadas:

```text
data/05_abt/abt_contratacao_binaria_geral.parquet
data/05_abt/abt_contratacao_ternaria_geral.parquet
data/05_abt/abt_contratacao_binaria_medicina.parquet
data/05_abt/abt_contratacao_ternaria_medicina.parquet
```

### 8.1 ABT binária

Usada nos modelos de contratação efetiva.

```text
Target binário:
0 = Não Contratado
1 = Contratada
```

Universo principal:

```text
Modalidade I
2019 a 2021
Situações consideradas: "Não Contratado" e "Contratada"
Unidade de análise: inscrição
```

### 8.2 ABT ternária

Usada nos modelos complementares com três situações do processo seletivo.

```text
Target ternário:
0 = Lista de Espera
1 = Não Contratado
2 = Contratada
```

Universo principal:

```text
Modalidade I
2019 a 2021
Situações consideradas: "Lista de Espera", "Não Contratado" e "Contratada"
Unidade de análise: inscrição
```

### 8.3 Variáveis principais

As variáveis centrais usadas nos modelos são:

```text
renda_familiar_per_capita
gap_nota_corte
interacao_renda_gap
```

### 8.4 Variáveis de controle

As variáveis de controle incluem características da inscrição, do curso e do processo seletivo, como:

```text
ano
semestre
região da instituição
turno
área CINE
subárea CINE
conceito do curso
opção de curso
demais variáveis disponíveis e utilizadas no artigo
```

### 8.5 Observação sobre vazamento de informação

O percentual de financiamento não deve ser usado como variável explicativa nos modelos de contratação, pois essa informação é observada apenas entre contratos efetivados. Seu uso como preditor poderia introduzir vazamento de informação.

Por isso, o percentual financiado é analisado separadamente na seção sobre renda, cobertura do financiamento e coparticipação implícita.

---

## 9. Modelos

O projeto utiliza modelos interpretáveis, compatíveis com o objetivo substantivo do artigo.

### 9.1 Modelos principais

```text
Regressão logística binária
Árvore de decisão binária
Regressão logística multinomial
Árvore de decisão ternária
```

Esses modelos são aplicados em dois recortes:

```text
1. Base geral da Modalidade I
2. Estudo de caso de Medicina
```

Assim, o apêndice de modelos pode conter oito tabelas:

```text
Tabela C1: Geral, target binário, regressão logística
Tabela C2: Geral, target binário, árvore de decisão
Tabela C3: Geral, target ternário, regressão logística multinomial
Tabela C4: Geral, target ternário, árvore de decisão
Tabela C5: Medicina, target binário, regressão logística
Tabela C6: Medicina, target binário, árvore de decisão
Tabela C7: Medicina, target ternário, regressão logística multinomial
Tabela C8: Medicina, target ternário, árvore de decisão
```

Cada tabela de experimento deve conter, no mínimo:

```text
experimento
recorte
target
modelo
tamanho_da_amostra
roc_auc
renda
desempenho_gap
interacao_renda_gap
quantidade_de_variaveis
observacoes
```

Para regressões logísticas, as colunas `renda`, `desempenho_gap` e `interacao_renda_gap` podem registrar coeficientes padronizados ou valores absolutos dos coeficientes, conforme definido no método.

Para árvores de decisão, essas colunas podem registrar importâncias das variáveis ou indicar a posição das variáveis nas primeiras divisões da árvore. A escolha deve ser documentada no artigo e nas notas das tabelas.

### 9.2 Por que não usar Random Forest como modelo principal

Random Forest pode melhorar desempenho preditivo, mas reduz a transparência da interpretação. Como o objetivo do artigo não é apenas prever contratação, mas entender como renda, desempenho e interação se associam à conversão em contrato, os modelos principais devem ser interpretáveis.

Por isso, Random Forest não é necessário como modelo principal. Caso seja usado, deve aparecer apenas como teste complementar de robustez preditiva, e não como base da interpretação substantiva.

### 9.3 Usar modelos salvos

Para gerar tabelas e figuras a partir dos modelos já ajustados:

```bash
python3 main.py modeling predict --use-saved-models
```

Para gerar as saídas do artigo usando os modelos salvos:

```bash
python3 main.py article modelos --use-saved-models
```

### 9.4 Reajustar modelos

Para refazer todos os modelos:

```bash
python3 main.py modeling fit all --force
```

Para refazer apenas os modelos gerais:

```bash
python3 main.py modeling fit general --force
```

Para refazer apenas os modelos de Medicina:

```bash
python3 main.py modeling fit medicina --force
```

Saídas esperadas:

```text
models/general/logit_binario.pkl
models/general/arvore_binaria.pkl
models/general/logit_multinomial.pkl
models/general/arvore_ternaria.pkl
models/medicina/logit_binario.pkl
models/medicina/arvore_binaria.pkl
models/medicina/logit_multinomial.pkl
models/medicina/arvore_ternaria.pkl
```

---

## 10. Geração das análises do artigo

Após a etapa de curadoria, é possível gerar cada parte do artigo separadamente.

### 10.1 Fluxo do processo seletivo

Gera as figuras da subseção 4.1 e os gráficos suplementares do Apêndice A.

```bash
python3 main.py article fluxo
```

Saídas esperadas:

```text
reports/article/figures/figura_1_fluxo_area_cine.png
reports/article/figures/figura_2_taxa_conversao_saude.png
reports/article/appendix/figura_A1_fluxo_curso_priorizado.png
reports/article/appendix/figura_A2_conversao_saude_priorizado.png
reports/article/appendix/figura_A3_fluxo_regiao.png
```

### 10.2 Matrizes por renda e desempenho

Gera a Tabela 1, a Figura 3 e os resultados suplementares do Apêndice B.

```bash
python3 main.py article matrizes
```

Saídas esperadas:

```text
reports/article/tables/tabela_1_situacao_inscricao.csv
reports/article/figures/figura_3_matrizes_renda_gap.png
reports/article/appendix/tabela_B1_faixas_renda.csv
reports/article/appendix/figura_B1_lista_espera_renda_gap.png
```

### 10.3 Renda, financiamento e coparticipação implícita

Gera a Figura 4, a Tabela 2, a correlação de Pearson e a regressão linear simples entre renda familiar per capita e percentual financiado entre contratos efetivados.

```bash
python3 main.py article financiamento
```

Saídas esperadas:

```text
reports/article/figures/figura_4_renda_percentual_financiado.png
reports/article/tables/tabela_2_correlacao_regressao_linear.csv
reports/article/tables/tabela_2_regressao_linear.csv
```

### 10.4 Modelos gerais

Gera a Figura 5, a Tabela 3 e as tabelas complementares dos modelos gerais.

```bash
python3 main.py article modelos-gerais --use-saved-models
```

Saídas esperadas:

```text
reports/article/figures/figura_5_probabilidade_contratacao.png
reports/article/tables/tabela_3_coeficientes_logistico.csv
reports/article/appendix/tabela_C1_geral_binario_logit.csv
reports/article/appendix/tabela_C2_geral_binario_arvore.csv
reports/article/appendix/tabela_C3_geral_ternario_logit_multinomial.csv
reports/article/appendix/tabela_C4_geral_ternario_arvore.csv
```

### 10.5 Estudo de caso de Medicina

Gera as figuras, tabelas e modelos específicos da subseção 4.5.

```bash
python3 main.py article medicina --use-saved-models
```

Saídas esperadas:

```text
reports/article/figures/figura_6_medicina_fluxo.png
reports/article/figures/figura_7_medicina_probabilidade.png
reports/article/tables/tabela_4_medicina_descritiva.csv
reports/article/appendix/tabela_C5_medicina_binario_logit.csv
reports/article/appendix/tabela_C6_medicina_binario_arvore.csv
reports/article/appendix/tabela_C7_medicina_ternario_logit_multinomial.csv
reports/article/appendix/tabela_C8_medicina_ternario_arvore.csv
```

### 10.6 Apêndices

Gera todas as tabelas e figuras suplementares.

```bash
python3 main.py article apendices --use-saved-models
```

Saídas esperadas:

```text
reports/article/appendix/
```

### 10.7 Todas as análises do artigo

Gera todas as tabelas e figuras do artigo a partir da base curada e dos modelos salvos.

```bash
python3 main.py article all --use-saved-models
```

Para gerar todas as tabelas e figuras reajustando os modelos:

```bash
python3 main.py article all --refit-models
```

---

## 11. Validações e rastreabilidade

O pipeline executa verificações de consistência entre as etapas, incluindo:

```text
preservação das chaves de inscrição e oferta
contagem de registros por etapa
identificação de duplicatas exatas
checagem de valores ausentes em variáveis centrais
verificação dos cruzamentos FIES × INEP
verificação da classificação da modalidade
checagem das bases usadas nas figuras e tabelas
```

Os relatórios de validação são salvos em:

```text
reports/diagnostics/
reports/logs/
```

Para executar os diagnósticos:

```bash
python3 main.py diagnostics all
```

---

## 12. Saídas esperadas do artigo

As principais saídas reproduzidas pelo pipeline são:

```text
Figura 1: Evolução quantitativa do fluxo do processo seletivo por área CINE
Figura 2: Taxa de conversão na área de Saúde e bem-estar
Tabela 1: Distribuição das inscrições por situação
Figura 3: Matrizes de contratação e não contratação por renda e desempenho
Figura 4: Associação entre renda familiar per capita e percentual financiado
Tabela 2: Correlação de Pearson e regressão linear simples
Figura 5: Probabilidades estimadas pelo modelo logístico
Tabela 3: Coeficientes principais do modelo logístico
Figura 6: Resultados do estudo de caso de Medicina
Tabela 4: Resultados descritivos de Medicina
Apêndice A: análises suplementares do fluxo de seleção
Apêndice B: análises suplementares por renda e desempenho
Apêndice C: análises complementares dos modelos binários e ternários
```

---

## 13. Reprodutibilidade

Para garantir reprodutibilidade, recomenda-se registrar:

```text
versão do Python
versões das bibliotecas em requirements.txt
data de download dos microdados
checksums dos arquivos brutos
contagem de registros por etapa
parâmetros dos modelos
semente aleatória usada nos modelos
versão dos modelos salvos
```

Quando houver aleatoriedade, os scripts devem utilizar semente fixa definida em `src/config.py`.

Exemplo:

```python
RANDOM_STATE = 42
```

---

## 14. Limitações dos dados e da modelagem

O percentual de financiamento é observado apenas para contratos efetivados. Por isso, a análise de renda e percentual financiado descreve a cobertura entre contratos formalizados, mas não observa diretamente a coparticipação enfrentada por candidatos que não chegaram à contratação.

O modelo logístico e a árvore de decisão identificam associações e padrões de classificação. Os resultados não devem ser interpretados como estimativas causais do efeito da coparticipação sobre a não contratação.

A árvore de decisão é utilizada por sua interpretabilidade, não como modelo de maior desempenho preditivo. O foco do artigo é compreender fatores associados à conversão em contrato, e não maximizar acurácia preditiva.

---

## 15. Arquivos não versionados

Por padrão, recomenda-se não versionar os microdados brutos, bases intermediárias grandes e saídas pesadas.

Exemplo de `.gitignore`:

```gitignore
data/01_raw/
data/02_staging/
data/03_interim/
data/04_curated/
data/05_abt/
reports/logs/
reports/diagnostics/
__pycache__/
.venv/
```

Os modelos salvos podem ser versionados se forem pequenos. Caso sejam grandes, recomenda-se usar Git LFS, release externa ou instruções para refazer os modelos localmente.

---

## 16. Autoria

Projeto desenvolvido por Lucas Ferreira Dias no âmbito de pesquisa acadêmica na Universidade Tecnológica Federal do Paraná, Campus Campo Mourão.

Orientação: Prof. Dr. André Luis Schwerz.

---

## 17. Licença e uso dos dados

Os microdados originais utilizados neste projeto são públicos e disponibilizados pelo MEC e pelo INEP. Este repositório disponibiliza apenas o código necessário para reproduzir o tratamento e as análises. Os dados brutos devem ser obtidos diretamente nas fontes oficiais.

O uso deste repositório é destinado a fins acadêmicos, científicos e de reprodutibilidade da pesquisa.