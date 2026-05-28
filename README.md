# FIES 2019–2021: elegibilidade acadêmica e contratação

Este repositório reúne os códigos utilizados para processar os microdados, gerar as bases analíticas, estimar os modelos e exportar as tabelas e figuras do artigo:

**Elegibilidade acadêmica e contratação no FIES: fatores associados à conversão em contrato no grupo de baixa renda, 2019–2021**

O estudo analisa a conversão entre elegibilidade acadêmica e contratação efetiva no Fundo de Financiamento Estudantil (FIES), com foco nos processos seletivos de 2019 a 2021.

Repositório: <https://github.com/eq-lucas/federal-microdata-kdd>

## Fontes de dados

Os dados brutos utilizados no estudo são públicos e devem ser baixados diretamente das fontes oficiais:

- [Microdados do FIES](https://dadosabertos.mec.gov.br/fies)
- [Microdados do Censo da Educação Superior](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior)

Para reproduzir os resultados, os arquivos brutos e arquivos brutos comapctados extraidos, devem ser colocados em `data/01_raw/`, preservando a estrutura esperada pelo pipeline.

Os microdados do FIES são a principal fonte da análise. Os microdados do Censo da Educação Superior são utilizados como fonte complementar para informações institucionais e de cursos.

## Estrutura do repositório

```text
data/
  01_raw/       dados brutos baixados das fontes oficiais
  02_staging/   dados brutos padronizados
  03_interim/   bases intermediárias
  04_curated/   bases curadas
  05_analysis/  bases analíticas descritivas
  06_abt/       bases analíticas de modelagem

models/         modelos treinados
reports/        logs, diagnósticos e produtos intermediários
article/        pacote final de tabelas, figuras e arquivos usados no artigo
src/            código-fonte do pipeline, análises, modelagem e exportação
main.py         ponto de entrada para execução das etapas
```

A pasta `article/` reúne os artefatos finais usados na redação do manuscrito, incluindo tabelas, figuras, arquivos CSV, PDFs, PNGs e arquivos TeX/LaTeX dos resultados e apêndices.

## Ambiente

Crie e ative um ambiente virtual Python:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reprodução completa

Para reproduzir os produtos finais usando os modelos já existentes:

```bash
python3 main.py reproduce all
```

Para refazer também o treinamento dos modelos:

```bash
python3 main.py reproduce all --refit --avaliacao in_sample
```

Para exportar novamente apenas o pacote final `article/`, a partir dos produtos já gerados em `reports/`:

```bash
python3 main.py export article --avaliacao in_sample --clean
```

## Etapas do processamento

### 1. Pipeline de dados

```bash
python3 main.py pipeline all
```

Executa a preparação dos dados brutos, as transformações intermediárias e a curadoria das bases.

### 2. Bases analíticas descritivas

```bash
python3 main.py analysis all
```

Gera as bases usadas nas análises descritivas de fluxo, taxas e distribuição das situações de inscrição.

### 3. Produtos descritivos do artigo

```bash
python3 main.py article all
```

Gera as tabelas e figuras descritivas usadas nas primeiras seções dos resultados.

Também é possível gerar produtos específicos:

```bash
python3 main.py article fluxo
python3 main.py article taxas-conversao
python3 main.py article tabelas-distribuicao
python3 main.py article matrizes
python3 main.py article financiamento
```

### 4. Bases analíticas de modelagem

```bash
python3 main.py abt binaria
python3 main.py abt ternaria --recorte geral
python3 main.py abt ternaria --recorte medicina
```

Esses comandos geram as bases usadas nos modelos do recorte geral e do recorte de Medicina.

### 5. Modelagem

```bash
python3 main.py modeling logit --force
python3 main.py modeling tree-depth --force
```

O primeiro comando estima os modelos logísticos. O segundo estima as árvores de decisão usadas como análise complementar.

### 6. Produtos de modelagem

```bash
python3 main.py article modelagem
```

Gera as tabelas, figuras e apêndices derivados dos modelos.

## Fluxo recomendado

Para refazer todos os resultados a partir dos dados brutos:

```bash
python3 main.py pipeline all
python3 main.py analysis all
python3 main.py article all
python3 main.py abt binaria
python3 main.py abt ternaria --recorte geral
python3 main.py abt ternaria --recorte medicina
python3 main.py modeling logit --force
python3 main.py modeling tree-depth --force
python3 main.py article modelagem
python3 main.py export article --avaliacao in_sample --clean
```

Para refazer apenas o pacote final `article/`, usando os produtos já existentes em `reports/`:

```bash
python3 main.py export article --avaliacao in_sample --clean
```

Para refazer apenas as tabelas de distribuição, incluindo a Tabela B1:

```bash
python3 main.py article tabelas-distribuicao
python3 main.py export article --avaliacao in_sample --clean
```

Para refazer apenas os produtos de modelagem e exportar novamente o pacote final:

```bash
python3 main.py modeling tree-depth --force
python3 main.py article modelagem
python3 main.py export article --avaliacao in_sample --clean
```

## Organização do pacote final

```text
article/secao_resultados_e_discussoes/
  4_1_fluxo_selecao/
  4_2_distribuicao_situacoes_renda_desempenho/
  4_3_financiamento_coparticipacao/
  4_4_logit_ternario_geral/
  4_5_medicina/

article/apendices/
  apendice_a_fluxo_selecao/
  apendice_b_desfechos_renda_desempenho/
  apendice_c_modelagem_geral/
  apendice_d_modelagem_medicina/
```

## Observação sobre os dados

Os dados brutos não são redistribuídos neste repositório. Para reproduzir os resultados, é necessário baixá-los diretamente das fontes oficiais indicadas acima e posicioná-los em `data/01_raw/`.

## Disponibilidade dos códigos

Os códigos deste repositório documentam o processamento, a análise e a geração dos artefatos utilizados no artigo. A pasta `article/` contém os produtos finais empregados na redação do manuscrito.
