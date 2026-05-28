"""
Organiza o pacote final de artefatos do artigo em article/.

O export final mantém apenas os arquivos que entram no artigo ou nos apêndices:
- figuras/tabelas descritivas das seções 4.1, 4.2 e 4.3;
- duas figuras principais do modelo multinomial geral na seção 4.4;
- duas figuras principais do modelo multinomial de Medicina na seção 4.5;
- Apêndice A e B descritivos;
- Apêndice C com tabelas de modelagem geral: logit ternário + árvore;
- Apêndice D com tabelas de modelagem de Medicina: logit ternário + árvore.

Não copia:
- C1/C2 antigas;
- tabelas delta do multinomial;
- figuras delta;
- figuras de árvores;
- coeficientes multinomiais para o corpo do artigo.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np


try:
    from src.constants import ROOT_DIR, REPORTS_DIR
except Exception:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    REPORTS_DIR = ROOT_DIR / "reports"


REPORTS_ARTICLE_DIR = REPORTS_DIR / "article"
DIAGNOSTICS_MODELING_DIR = REPORTS_DIR / "diagnostics" / "modeling"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "article"

EXTENSOES_EXPORTADAS = {".pdf", ".png", ".csv", ".tex", ".latex"}

CLASSE_LABEL = {
    0: "Lista de espera",
    1: "Não contratado",
    2: "Contratada",
}

LABEL_VARIAVEIS = {
    "renda_per_capita": "Renda familiar per capita",
    "gap": "Desempenho relativo à nota de corte",
    "renda_gap": "Interação renda × desempenho",
    "idade": "Idade",
    "nota_corte_gp": "Nota de corte do grupo",
    "opcao_curso": "Opção de curso",
    "ano": "Ano do processo seletivo",
    "semestre": "Semestre do processo seletivo",
    "conceito_curso_gp": "Conceito do curso",
    "turno": "Turno",
    "ensino_medio_publico": "Ensino médio público",
    "ensino_medio_escola_publica": "Ensino médio público",
    "subarea_conhecimento": "Subárea de conhecimento",
    "regiao_ies_alvo": "Região da oferta",
    "natureza_juridica_mantenedora": "Natureza jurídica da mantenedora",
    "etnia_cor": "Cor/raça ou etnia",
    "sexo": "Sexo",
    "regiao_morar": "Região de residência",
    "organizacao_academica": "Organização acadêmica",
    "concluiu_curso_superior": "Concluiu curso superior",
    "beneficiado_creduc_fies": "Beneficiado CREDUC/FIES",
    "uf_local_oferta": "UF do local de oferta",
}


@dataclass(frozen=True)
class ItemCopia:
    descricao: str
    destino: str
    padroes: tuple[str, ...]
    nome_base: str | None = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _limpar_pasta(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _fmt_int_br(valor) -> str:
    if pd.isna(valor):
        return ""
    return f"{int(round(float(valor))):,}".replace(",", ".")


def _fmt_float_br(valor, casas: int = 4) -> str:
    if pd.isna(valor):
        return ""
    return f"{float(valor):.{casas}f}".replace(".", ",")


def _latex_escape(valor: object) -> str:
    texto = "" if pd.isna(valor) else str(valor)
    trocas = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for a, b in trocas.items():
        texto = texto.replace(a, b)
    return texto


def _glob_many(base: Path, padroes: Iterable[str]) -> list[Path]:
    arquivos: list[Path] = []
    for padrao in padroes:
        arquivos.extend(base.glob(padrao))
    arquivos = sorted(set(p for p in arquivos if p.is_file()))
    return [p for p in arquivos if p.suffix.lower() in EXTENSOES_EXPORTADAS]


def _copy_one(origem: Path, destino_dir: Path, nome_base: str | None = None) -> Path:
    _ensure_dir(destino_dir)
    destino = destino_dir / (f"{nome_base}{origem.suffix.lower()}" if nome_base else origem.name)
    shutil.copy2(origem, destino)
    return destino


def copiar_item(item: ItemCopia, output_dir: Path, dry_run: bool = False) -> list[Path]:
    arquivos = _glob_many(REPORTS_ARTICLE_DIR, item.padroes)
    destino_dir = output_dir / item.destino

    if dry_run:
        print(f"[DRY] {item.descricao}: {len(arquivos)} arquivo(s)")
        for p in arquivos:
            print(f"      {p}")
        return []

    saidas = [_copy_one(p, destino_dir, item.nome_base) for p in arquivos]
    print(f"[OK] {item.descricao}: {len(saidas)} arquivo(s)")
    return saidas


def salvar_tabela_cinza(
    df: pd.DataFrame,
    destino_dir: Path,
    nome_base: str,
    caption: str,
    label: str,
    notas: str | None = None,
    gerar_imagem: bool = True,
) -> None:
    _ensure_dir(destino_dir)

    df_out = df.copy()
    df_out.to_csv(destino_dir / f"{nome_base}.csv", index=False)

    linhas = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{'l' * len(df_out.columns)}}}",
        r"\toprule",
        r"\rowcolor[gray]{0.85}",
        " & ".join(_latex_escape(c) for c in df_out.columns) + r" \\",
        r"\midrule",
    ]

    for i, (_, row) in enumerate(df_out.iterrows()):
        if i % 2 == 1:
            linhas.append(r"\rowcolor[gray]{0.95}")
        linhas.append(" & ".join(_latex_escape(v) for v in row.tolist()) + r" \\")

    linhas += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    if notas:
        linhas.append(rf"\par\footnotesize{{Notas: {_latex_escape(notas)}}}")

    linhas.append(r"\par\footnotesize{Fonte: elaboração própria (2026).}")
    linhas.append(r"\end{table}")

    tex = "\n".join(linhas) + "\n"
    (destino_dir / f"{nome_base}.tex").write_text(tex, encoding="utf-8")
    (destino_dir / f"{nome_base}.latex").write_text(tex, encoding="utf-8")

    if not gerar_imagem:
        print(f"[OK] Tabela gerada: {destino_dir / nome_base}.[csv,tex,latex]")
        return

    fig_h = max(1.2, 0.42 * (len(df_out) + 1))
    fig_w = max(7.0, 1.55 * len(df_out.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df_out.values,
        colLabels=df_out.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#4d4d4d")
        cell.set_linewidth(0.45)

        if row == 0:
            cell.set_facecolor("#d9d9d9")
            cell.set_text_props(weight="bold", color="black")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")
        else:
            cell.set_facecolor("#ffffff")

        if col == 0 and row > 0:
            cell.set_text_props(weight="bold")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(destino_dir / f"{nome_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(destino_dir / f"{nome_base}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] Tabela gerada: {destino_dir / nome_base}.*")


def _find_csv_contains(partes_obrigatorias: Sequence[str], final: str) -> Path:
    candidatos = sorted(DIAGNOSTICS_MODELING_DIR.rglob(final))
    for path in candidatos:
        s = str(path).lower()
        if all(parte.lower() in s for parte in partes_obrigatorias):
            return path
    raise FileNotFoundError(f"Não encontrei {final} com partes obrigatórias: {partes_obrigatorias}")


def _col(df: pd.DataFrame, nomes: Sequence[str]) -> str:
    for nome in nomes:
        if nome in df.columns:
            return nome
    raise KeyError(f"Nenhuma coluna encontrada: {nomes}")


def copiar_elementos_fixos(output_dir: Path, dry_run: bool = False) -> None:
    itens = [
        ItemCopia(
            "Figura 1 - fluxo por áreas CINE",
            "secao_resultados_e_discussoes/4_1_fluxo_selecao/figura_1",
            ("figures/fluxo_selecao/funil/figura_1_funil_fies_inscritos_area_cine.*",),
            "figura_1_funil_fies_inscritos_area_cine",
        ),
        ItemCopia(
            "Figura 2 - taxa de conversão Saúde e bem-estar",
            "secao_resultados_e_discussoes/4_1_fluxo_selecao/figura_2",
            ("figures/taxas_conversao/taxa_conversao_inscritos/grafico_taxa_conversao_inscritos_saude_e_bem_estar.*",),
            "figura_2_taxa_conversao_inscritos_saude_e_bem_estar",
        ),
        ItemCopia(
            "Tabela 1 - distribuição das situações",
            "secao_resultados_e_discussoes/4_2_distribuicao_situacoes_renda_desempenho/tabela_1",
            ("tables/secao_4_2/tabela_1_distribuicao_inscricoes_por_situacao.*",),
            "tabela_1_distribuicao_inscricoes_por_situacao",
        ),
        ItemCopia(
            "Figura 3 - matrizes contratação e não contratação",
            "secao_resultados_e_discussoes/4_2_distribuicao_situacoes_renda_desempenho/figura_3",
            ("figures/heatmap_contratados_x_nao_contratados/nacional_contratados_nao_contratados.*",),
            "figura_3_contratados_nao_contratados_renda_desempenho",
        ),
        ItemCopia(
            "Figura 4 - renda e percentual financiado",
            "secao_resultados_e_discussoes/4_3_financiamento_coparticipacao/figura_4",
            ("figures/financiamento_coparticipacao/figura_4_associacao_renda_percentual_financiamento.*",),
            "figura_4_associacao_renda_percentual_financiamento",
        ),
        ItemCopia(
            "Tabela 2 - renda e financiamento",
            "secao_resultados_e_discussoes/4_3_financiamento_coparticipacao/tabela_2",
            (
                "tables/secao_4_3/tabela_2_associacao_renda_financiamento.*",
                "tables/secao_4_3/regressao_renda_financiamento_resultados.csv",
                "tables/secao_4_3/dados_figura_4_renda_financiamento.csv",
            ),
        ),
        ItemCopia(
            "Figura A1 - fluxo curso priorizado",
            "apendices/apendice_a_fluxo_selecao/figura_a1",
            ("figures/fluxo_selecao/funil/figura_1a_funil_fies_candidatos_unicos_area_cine.*",),
            "figura_a1_funil_curso_priorizado_area_cine",
        ),
        ItemCopia(
            "Figura A2 - taxa curso priorizado Saúde e bem-estar",
            "apendices/apendice_a_fluxo_selecao/figura_a2",
            ("figures/taxas_conversao/taxa_conversao_curso_priorizado/grafico_taxa_conversao_curso_priorizado_saude_e_bem_estar.*",),
            "figura_a2_taxa_conversao_curso_priorizado_saude_e_bem_estar",
        ),
        ItemCopia(
            "Figura A3 - fluxo por região",
            "apendices/apendice_a_fluxo_selecao/figura_a3",
            ("figures/fluxo_selecao/funil/funil_fies_inscritos_regiao_total.*",),
            "figura_a3_funil_regiao_total",
        ),
        ItemCopia(
            "Tabela B1 - distribuição por faixa de renda",
            "apendices/apendice_b_desfechos_renda_desempenho/tabela_b1",
            ("appendix/apendice_b/tabela_b1_distribuicao_inscricoes_por_faixa_renda.*",),
            "tabela_b1_distribuicao_inscricoes_por_faixa_renda",
        ),
        ItemCopia(
            "Figura B1 - lista de espera por renda e desempenho",
            "apendices/apendice_b_desfechos_renda_desempenho/figura_b1",
            ("appendix/apendice_b/heatmap_lista_espera/figura_b1_lista_espera.*",),
            "figura_b1_lista_espera_renda_desempenho",
        ),
    ]

    for item in itens:
        copiar_item(item, output_dir, dry_run=dry_run)

    if not dry_run:
        corrigir_tabela_b1(output_dir)


def _parse_int_br(valor) -> float:
    """Converte inteiros no padrão brasileiro para número.

    Exemplos:
    - "450.100" -> 450100
    - "49.450" -> 49450
    - "1.925.666" -> 1925666

    Não usa pd.to_numeric diretamente porque o ponto, neste caso, é separador
    de milhar, não separador decimal.
    """
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    texto = texto.replace("%", "")
    texto = texto.replace(".", "")
    texto = texto.replace(",", ".")
    try:
        return float(texto)
    except Exception:
        return np.nan


def _fmt_percentual_br(valor, casas: int = 2) -> str:
    if pd.isna(valor):
        return ""
    return f"{float(valor):.{casas}f}%".replace(".", ",")


def corrigir_tabela_b1(output_dir: Path) -> None:
    pasta = output_dir / "apendices" / "apendice_b_desfechos_renda_desempenho" / "tabela_b1"
    csv_path = pasta / "tabela_b1_distribuicao_inscricoes_por_faixa_renda.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path, dtype=str)
    col_faixa = df.columns[0]
    col_n = df.columns[1]

    ordem = {
        "Até 600": 0,
        "Ate 600": 0,
        "601–1.200": 1,
        "601-1.200": 1,
        "1.201–1.800": 2,
        "1.201-1.800": 2,
        "1.801–2.400": 3,
        "1.801-2.400": 3,
        "2.401–3.000": 4,
        "2.401-3.000": 4,
        "Acima de 3.000": 5,
    }

    df["_ordem"] = df[col_faixa].map(lambda x: ordem.get(str(x).strip(), 999))
    df = df.sort_values("_ordem").drop(columns="_ordem").reset_index(drop=True)

    inscricoes_num = df[col_n].map(_parse_int_br)
    total = float(inscricoes_num.sum())
    if total <= 0:
        raise ValueError(f"Tabela B1 sem total válido de inscrições: {csv_path}")

    perc = inscricoes_num / total * 100
    perc_acum = perc.cumsum()

    # Preserva a contagem no formato brasileiro de milhares e recalcula os
    # percentuais depois de ordenar as faixas do menor para o maior valor.
    df[col_n] = inscricoes_num.map(_fmt_int_br)
    df["%"] = perc.map(lambda x: _fmt_percentual_br(x, 2))
    df["% acumulado"] = perc_acum.map(lambda x: _fmt_percentual_br(x, 2))

    salvar_tabela_cinza(
        df,
        pasta,
        "tabela_b1_distribuicao_inscricoes_por_faixa_renda",
        "Distribuição das inscrições por faixa de renda familiar per capita no FIES, 2019–2021.",
        "tab:tabela_b1_distribuicao_inscricoes_por_faixa_renda",
        notas="Percentuais recalculados após ordenação crescente das faixas de renda.",
    )


def gerar_efeitos_multinomiais_se_possivel(recorte: str, avaliacao: str) -> None:
    try:
        from src.article import efeitos_multinomiais_ternario
        efeitos_multinomiais_ternario.run(recorte=recorte, avaliacao=avaliacao, experimento="E5")
    except Exception as exc:
        print(f"[AVISO] Não foi possível gerar efeitos multinomiais para recorte={recorte}, avaliação={avaliacao}: {exc}")


def copiar_efeitos_multinomiais(recorte: str, avaliacao: str, output_dir: Path, dry_run: bool = False) -> None:
    secao = "4_4_logit_ternario_geral" if recorte == "geral" else "4_5_medicina"
    figura_pasta = f"figures/efeitos_multinomiais_recorte_{recorte}"

    copiar_item(
        ItemCopia(
            descricao=f"Efeitos multinomiais - figuras principais - {recorte}",
            destino=f"secao_resultados_e_discussoes/{secao}/efeitos_multinomiais",
            padroes=(
                f"{figura_pasta}/figura_probabilidade_por_desempenho_todas_classes_logit_ternario_recorte_{recorte}_{avaliacao}_e5.*",
                f"{figura_pasta}/figura_probabilidade_por_renda_todas_classes_logit_ternario_recorte_{recorte}_{avaliacao}_e5.*",
            ),
        ),
        output_dir,
        dry_run=dry_run,
    )


def _carregar_metricas_logit_ternario(recorte: str, avaliacao: str) -> pd.DataFrame:
    path = _find_csv_contains(["logit_ternario", avaliacao, recorte], "logit_ternario_experimentos_metricas.csv")
    df = pd.read_csv(path)

    if "recorte" in df.columns:
        df = df[df["recorte"].astype(str).str.contains(recorte, case=False, na=False)].copy()
    if "avaliacao" in df.columns:
        df = df[df["avaliacao"].astype(str).eq(avaliacao)].copy()

    return df


def gerar_tabela_compacta_logit_ternario(recorte: str, avaliacao: str, output_dir: Path) -> None:
    df = _carregar_metricas_logit_ternario(recorte, avaliacao)

    exp_col = _col(df, ["experimento", "especificacao"])
    n_col = _col(df, ["n_teste", "n_total_abt", "n"])
    roc_col = "roc_auc_ovr_weighted" if "roc_auc_ovr_weighted" in df.columns else _col(df, ["roc_auc"])
    bal_col = "balanced_accuracy" if "balanced_accuracy" in df.columns else None
    f1_col = "f1_macro" if "f1_macro" in df.columns else None

    linhas = []
    for _, row in df.sort_values(exp_col).iterrows():
        exp_num = str(row[exp_col]).replace("E", "")
        linhas.append(
            {
                "Especificação": f"Modelo {exp_num}",
                "N": _fmt_int_br(row[n_col]),
                "ROC-AUC": _fmt_float_br(row[roc_col]),
                "Acurácia balanceada": _fmt_float_br(row[bal_col]) if bal_col else "",
                "F1 macro": _fmt_float_br(row[f1_col]) if f1_col else "",
                "Variáveis": _fmt_int_br(row.get("blocos_variaveis_qtd", row.get("colunas_finais_pos_processamento", pd.NA))),
            }
        )

    tabela = pd.DataFrame(linhas)
    destino = output_dir / "apendices" / (
        "apendice_c_modelagem_geral" if recorte == "geral" else "apendice_d_modelagem_medicina"
    )

    salvar_tabela_cinza(
        tabela,
        destino,
        f"tabela_compacta_logit_ternario_recorte_{recorte}_{avaliacao}",
        f"Resumo das especificações do modelo logístico multinomial, recorte {recorte}, avaliação {avaliacao}.",
        f"tab:tabela_compacta_logit_ternario_recorte_{recorte}_{avaliacao}",
        notas="As especificações E1 a E5 adicionam progressivamente controles observáveis antes do desfecho.",
    )


def _modelo_path(recorte: str, avaliacao: str, experimento: str) -> Path | None:
    grupo = "general" if recorte == "geral" else "medicina"
    base = ROOT_DIR / "models" / grupo / "logit_ternario" / avaliacao
    for name in [f"logit_ternario_{experimento.lower()}.joblib", f"logit_ternario_{experimento.upper()}.joblib"]:
        path = base / name
        if path.exists():
            return path
    achados = sorted(base.glob(f"*{experimento.lower()}*.joblib")) + sorted(base.glob(f"*{experimento.upper()}*.joblib"))
    return achados[0] if achados else None


def _estimador_final(modelo):
    if hasattr(modelo, "steps"):
        return modelo.steps[-1][1]
    return modelo


def _feature_names_pipeline(modelo) -> list[str] | None:
    if hasattr(modelo, "steps"):
        pre = modelo[:-1]
        try:
            return [str(x) for x in pre.get_feature_names_out()]
        except Exception:
            pass

    nomes = getattr(modelo, "feature_names_in_", None)
    if nomes is not None:
        return [str(x) for x in nomes]

    if hasattr(modelo, "named_steps"):
        for step in modelo.named_steps.values():
            nomes = getattr(step, "feature_names_in_", None)
            if nomes is not None:
                return [str(x) for x in nomes]

    return None


def _match_feature(feature_names: list[str], var: str) -> int | None:
    candidatos = []
    for i, name in enumerate(feature_names):
        clean = name.split("__")[-1]
        if clean == var or clean.endswith(var):
            candidatos.append(i)
    if candidatos:
        return candidatos[0]

    padrao = re.compile(rf"(^|__){re.escape(var)}($|[^A-Za-z0-9_])")
    for i, name in enumerate(feature_names):
        if padrao.search(name):
            return i

    return None


def gerar_tabela_coeficientes_logit_ternario(recorte: str, avaliacao: str, output_dir: Path) -> None:
    linhas = []

    for experimento in ["E1", "E2", "E3", "E4", "E5"]:
        path = _modelo_path(recorte, avaliacao, experimento)
        if path is None:
            continue

        try:
            modelo = joblib.load(path)
            est = _estimador_final(modelo)
            coefs = getattr(est, "coef_", None)
            classes = getattr(est, "classes_", [0, 1, 2])
            feature_names = _feature_names_pipeline(modelo)
        except Exception:
            continue

        if coefs is None or feature_names is None:
            continue

        classes = [int(c) for c in classes]

        for classe_idx, classe in enumerate(classes):
            if classe_idx >= coefs.shape[0]:
                continue

            row = {
                "Especificação": experimento.replace("E", "Modelo "),
                "Classe": CLASSE_LABEL.get(classe, str(classe)),
            }

            for var, label in [
                ("renda_per_capita", "Renda"),
                ("gap", "Desempenho"),
                ("renda_gap", "Interação"),
            ]:
                idx = _match_feature(feature_names, var)
                row[label] = _fmt_float_br(coefs[classe_idx, idx]) if idx is not None else ""

            linhas.append(row)

    if not linhas:
        print(f"[AVISO] Coeficientes do logit ternário não extraídos para recorte={recorte}, avaliação={avaliacao}.")
        return

    tabela = pd.DataFrame(linhas)
    destino = output_dir / "apendices" / (
        "apendice_c_modelagem_geral" if recorte == "geral" else "apendice_d_modelagem_medicina"
    )

    salvar_tabela_cinza(
        tabela,
        destino,
        f"tabela_coeficientes_chave_logit_ternario_recorte_{recorte}_{avaliacao}",
        f"Coeficientes das variáveis-chave no modelo logístico multinomial, recorte {recorte}, avaliação {avaliacao}.",
        f"tab:coeficientes_chave_logit_ternario_recorte_{recorte}_{avaliacao}",
        notas=(
            "Coeficientes extraídos dos modelos treinados. Em modelos multinomiais, os coeficientes pertencem às classes do desfecho; "
            "por isso, devem ser lidos como informação complementar às probabilidades previstas."
        ),
    )



def _diagnostics_tree_dir(target: str, recorte: str, avaliacao: str, profundidade: int) -> Path:
    """Diretório canônico dos diagnostics da árvore por profundidade."""
    return (
        DIAGNOSTICS_MODELING_DIR
        / f"treeClassification_{int(profundidade)}_profundidade_{target}"
        / avaliacao
        / recorte
    )


def _carregar_metricas_tree_profundidade(
    target: str,
    recorte: str,
    avaliacao: str,
    profundidade: int,
) -> pd.DataFrame:
    """Carrega a tabela de métricas diretamente dos diagnostics do modelo treinado."""
    path = (
        _diagnostics_tree_dir(target, recorte, avaliacao, profundidade)
        / f"treeClassification_{int(profundidade)}_profundidade_{target}_experimentos_metricas.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Métricas da árvore não encontradas: {path}. "
            f"Rode a modelagem correspondente antes do export."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Tabela de métricas vazia: {path}")

    if "target" in df.columns:
        df = df[df["target"].astype(str).str.lower().eq(target)].copy()
    if "recorte" in df.columns:
        df = df[df["recorte"].astype(str).str.lower().eq(recorte)].copy()
    if "avaliacao" in df.columns:
        df = df[df["avaliacao"].astype(str).eq(avaliacao)].copy()

    if df.empty:
        raise ValueError(
            f"Nenhuma métrica após filtros target={target}, recorte={recorte}, "
            f"avaliacao={avaliacao}, profundidade={profundidade}: {path}"
        )

    return df


def _carregar_importancias_agregadas_tree_profundidade(
    target: str,
    recorte: str,
    avaliacao: str,
    profundidade: int,
    experimento: str = "E5",
) -> pd.DataFrame:
    """Carrega somente importâncias agregadas por variável original.

    Usa exclusivamente *_importancias_agregadas.csv. Não mistura com
    *_importancias_transformadas.csv, porque isso duplicaria a decomposição de
    importância da mesma árvore. Para `DecisionTreeClassifier.feature_importances_`,
    a soma das importâncias normalizadas do experimento deve ser 1, salvo erro
    numérico mínimo ou árvore sem divisões.
    """
    path = (
        _diagnostics_tree_dir(target, recorte, avaliacao, profundidade)
        / f"treeClassification_{int(profundidade)}_profundidade_{target}_importancias_agregadas.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Importâncias agregadas da árvore não encontradas: {path}. "
            f"Rode a modelagem correspondente antes do export."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Tabela de importâncias vazia: {path}")

    if "target" in df.columns:
        df = df[df["target"].astype(str).str.lower().eq(target)].copy()
    if "recorte" in df.columns:
        df = df[df["recorte"].astype(str).str.lower().eq(recorte)].copy()
    if "avaliacao" in df.columns:
        df = df[df["avaliacao"].astype(str).eq(avaliacao)].copy()
    if "experimento" in df.columns:
        df = df[df["experimento"].astype(str).str.upper().eq(experimento.upper())].copy()

    if df.empty:
        raise ValueError(
            f"Nenhuma importância após filtros target={target}, recorte={recorte}, "
            f"avaliacao={avaliacao}, profundidade={profundidade}, experimento={experimento}: {path}"
        )

    var_col = _col(df, ["variavel_original", "variavel", "feature_original"])
    imp_col = _col(df, ["importancia_normalizada", "importancia", "importance"])

    df = df.copy()
    df[imp_col] = pd.to_numeric(df[imp_col], errors="coerce").fillna(0.0)

    # Se o arquivo só tiver importância bruta, normaliza localmente. Se já tiver
    # importancia_normalizada, preserva a escala e valida a soma.
    if imp_col != "importancia_normalizada":
        total_bruto = float(df[imp_col].sum())
        df["importancia_normalizada"] = df[imp_col] / total_bruto if total_bruto > 0 else 0.0
        imp_col = "importancia_normalizada"

    soma = float(df[imp_col].sum())
    if soma > 1.000001:
        raise ValueError(
            f"Importâncias normalizadas somam {soma:.6f} em {path}. "
            "Isso indica duplicação/agregação indevida; para feature_importances_ "
            "da árvore a soma por experimento não deve exceder 1."
        )

    df = df[[var_col, imp_col]].copy()
    df.columns = ["variavel_original", "importancia_normalizada"]
    return df


def _montar_tabela_compacta_tree(df: pd.DataFrame, target: str) -> pd.DataFrame:
    target = target.lower()
    exp_col = _col(df, ["experimento", "especificacao"])
    n_col = _col(df, ["n_total_abt", "n", "n_teste"])
    roc_col = "roc_auc" if target == "binario" and "roc_auc" in df.columns else (
        "roc_auc_ovr_weighted" if "roc_auc_ovr_weighted" in df.columns else _col(df, ["roc_auc"])
    )

    def first(row, names):
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return row[name]
        return pd.NA

    linhas = []
    for _, row in df.sort_values(exp_col).iterrows():
        exp_num = str(row[exp_col]).replace("E", "")
        linhas.append(
            {
                "Especificação": f"Modelo {exp_num}",
                "N": _fmt_int_br(row[n_col]),
                "ROC-AUC": _fmt_float_br(row[roc_col]),
                "Renda": _fmt_float_br(first(row, ["importancia_renda_per_capita", "imp_renda_per_capita"])),
                "Desempenho": _fmt_float_br(first(row, ["importancia_gap", "imp_gap"])),
                "Interação": _fmt_float_br(first(row, ["importancia_renda_gap", "imp_renda_gap"])),
                "Variáveis": _fmt_int_br(row.get("blocos_variaveis_qtd", row.get("colunas_finais_pos_processamento", pd.NA))),
            }
        )
    return pd.DataFrame(linhas)


def _montar_top10_importancias_tree(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("variavel_original", as_index=False)["importancia_normalizada"]
        .sum()
        .sort_values("importancia_normalizada", ascending=False)
        .head(10)
        .copy()
    )

    soma_top10 = float(agg["importancia_normalizada"].sum())
    if soma_top10 > 1.000001:
        raise ValueError(
            f"Top 10 de importâncias soma {soma_top10:.6f}. "
            "Isso não é coerente com importâncias normalizadas por árvore."
        )

    agg.insert(0, "Posição", range(1, len(agg) + 1))
    agg["Variável"] = agg["variavel_original"].map(lambda x: LABEL_VARIAVEIS.get(str(x), str(x)))
    agg["Importância"] = agg["importancia_normalizada"].map(lambda x: _fmt_float_br(x, 4))
    return agg[["Posição", "Variável", "Importância"]]


def gerar_tabela_compacta_tree_19_ternario(recorte: str, avaliacao: str, output_dir: Path) -> None:
    df = _carregar_metricas_tree_profundidade(
        target="ternario",
        recorte=recorte,
        avaliacao=avaliacao,
        profundidade=19,
    )
    tabela = _montar_tabela_compacta_tree(df, target="ternario")
    destino = output_dir / "apendices" / (
        "apendice_c_modelagem_geral" if recorte == "geral" else "apendice_d_modelagem_medicina"
    )

    salvar_tabela_cinza(
        tabela,
        destino,
        f"tabela_compacta_treeClassification_19_profundidade_ternario_recorte_{recorte}_{avaliacao}",
        f"Resumo da árvore de decisão multinomial, recorte {recorte}, avaliação {avaliacao}, profundidade máxima 19.",
        f"tab:tabela_compacta_treeClassification_19_profundidade_ternario_recorte_{recorte}_{avaliacao}",
        notas="Renda, desempenho e interação indicam importâncias normalizadas das variáveis na árvore. A profundidade máxima parametrizada é 19.",
        gerar_imagem=True,
    )


def gerar_top10_importancias_tree_19_ternario(recorte: str, avaliacao: str, output_dir: Path) -> None:
    df = _carregar_importancias_agregadas_tree_profundidade(
        target="ternario",
        recorte=recorte,
        avaliacao=avaliacao,
        profundidade=19,
        experimento="E5",
    )
    tabela = _montar_top10_importancias_tree(df)
    destino = output_dir / "apendices" / (
        "apendice_c_modelagem_geral" if recorte == "geral" else "apendice_d_modelagem_medicina"
    )

    salvar_tabela_cinza(
        tabela,
        destino,
        f"tabela_top10_importancias_treeClassification_19_ternario_recorte_{recorte}_{avaliacao}",
        f"Dez variáveis de maior importância na árvore de decisão multinomial, recorte {recorte}, avaliação {avaliacao}, E5.",
        f"tab:top10_importancias_treeClassification_19_ternario_{recorte}_{avaliacao}",
        notas="Importâncias normalizadas por variável original, calculadas a partir da importância por redução de impureza da árvore. Valores maiores indicam maior contribuição para as partições, sem interpretação causal.",
        gerar_imagem=True,
    )


def gerar_tabela_compacta_tree_10_binario_geral(avaliacao: str, output_dir: Path) -> None:
    """Gera a tabela compacta da árvore binária da etapa contratual, profundidade 10.

    Esta análise usa a ABT binária geral, isto é, compara somente inscrições em
    não contratação e contratação. Ela é exportada apenas para o Apêndice C.
    """
    df = _carregar_metricas_tree_profundidade(
        target="binario",
        recorte="geral",
        avaliacao=avaliacao,
        profundidade=10,
    )
    tabela = _montar_tabela_compacta_tree(df, target="binario")
    destino = output_dir / "apendices" / "apendice_c_modelagem_geral"

    salvar_tabela_cinza(
        tabela,
        destino,
        f"tabela_compacta_treeClassification_10_profundidade_binario_recorte_geral_{avaliacao}",
        f"Resumo da árvore de decisão da etapa contratual, recorte geral, avaliação {avaliacao}, profundidade máxima 10.",
        f"tab:tabela_compacta_treeClassification_10_profundidade_binario_recorte_geral_{avaliacao}",
        notas=(
            "A etapa contratual compara inscrições em não contratação e contratação. "
            "Renda, desempenho e interação indicam importâncias normalizadas das variáveis na árvore. "
            "A profundidade máxima parametrizada é 10."
        ),
        gerar_imagem=True,
    )


def gerar_top10_importancias_tree_10_binario_geral(avaliacao: str, output_dir: Path) -> None:
    """Gera top 10 de importâncias da árvore binária da etapa contratual, E5, profundidade 10."""
    df = _carregar_importancias_agregadas_tree_profundidade(
        target="binario",
        recorte="geral",
        avaliacao=avaliacao,
        profundidade=10,
        experimento="E5",
    )
    tabela = _montar_top10_importancias_tree(df)
    destino = output_dir / "apendices" / "apendice_c_modelagem_geral"

    salvar_tabela_cinza(
        tabela,
        destino,
        f"tabela_top10_importancias_treeClassification_10_binario_recorte_geral_{avaliacao}",
        f"Dez variáveis de maior importância na árvore de decisão da etapa contratual, recorte geral, avaliação {avaliacao}, E5, profundidade máxima 10.",
        f"tab:top10_importancias_treeClassification_10_binario_geral_{avaliacao}",
        notas=(
            "A etapa contratual compara inscrições em não contratação e contratação. "
            "Importâncias normalizadas por variável original, calculadas a partir da importância por redução de impureza da árvore; "
            "valores maiores indicam maior contribuição para as partições, sem interpretação causal."
        ),
        gerar_imagem=True,
    )


def gerar_tabelas_tree_10_binario_geral(avaliacao: str, output_dir: Path) -> None:
    try:
        gerar_tabela_compacta_tree_10_binario_geral(avaliacao, output_dir)
    except Exception as exc:
        print(f"[AVISO] Tabela compacta da árvore binária contratual não exportada: {exc}")

    try:
        gerar_top10_importancias_tree_10_binario_geral(avaliacao, output_dir)
    except Exception as exc:
        print(f"[AVISO] Top 10 da árvore binária contratual não exportado: {exc}")


def _destino_modelagem(recorte: str) -> Path:
    return Path("apendices/apendice_c_modelagem_geral" if recorte == "geral" else "apendices/apendice_d_modelagem_medicina")


def _base_tree_article(recorte: str, avaliacao: str) -> Path:
    return (
        REPORTS_ARTICLE_DIR
        / "appendix"
        / "apendice_modelagem"
        / f"treeClassification_19_profundidade_ternario_recorte_{recorte}"
        / avaliacao
        / recorte
    )


def _ler_tabela_csv_ou_tex(path_base: Path) -> pd.DataFrame | None:
    """Tenta ler uma tabela a partir de CSV. Se só houver TEX/LATEX, tenta extrair o tabular simples."""
    csv_path = path_base.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    for ext in [".tex", ".latex"]:
        tex_path = path_base.with_suffix(ext)
        if not tex_path.exists():
            continue

        texto = tex_path.read_text(encoding="utf-8", errors="ignore")
        linhas = []
        dentro = False

        for raw in texto.splitlines():
            linha = raw.strip()
            if linha.startswith(r"\begin{tabular}"):
                dentro = True
                continue
            if linha.startswith(r"\end{tabular}"):
                break
            if not dentro:
                continue
            if (
                not linha
                or linha.startswith(r"\toprule")
                or linha.startswith(r"\midrule")
                or linha.startswith(r"\bottomrule")
                or linha.startswith(r"\rowcolor")
                or linha.startswith("\\")
            ):
                continue
            if "&" not in linha:
                continue
            linha = linha.replace(r"\\", "").strip()
            partes = [p.strip().replace(r"\_", "_") for p in linha.split("&")]
            linhas.append(partes)

        if len(linhas) >= 2:
            header = linhas[0]
            data = [row for row in linhas[1:] if len(row) == len(header)]
            if data:
                return pd.DataFrame(data, columns=header)

    return None


def _candidatos_tabela_compacta_tree(recorte: str, avaliacao: str) -> list[Path]:
    base_exata = _base_tree_article(recorte, avaliacao)
    stems = [
        base_exata / f"tabela_compacta_treeClassification_19_profundidade_ternario_recorte_{recorte}_{avaliacao}",
        base_exata / f"tabela_compacta_treeClassification_19_profundidade_ternario_{recorte}_{avaliacao}",
    ]

    candidatos: list[Path] = []
    for stem in stems:
        for ext in [".csv", ".tex", ".latex"]:
            p = stem.with_suffix(ext)
            if p.exists():
                candidatos.append(stem)
                break

    for p in REPORTS_ARTICLE_DIR.glob(
        f"appendix/apendice_modelagem/**/tabela_compacta_treeClassification*19*ternario*recorte_{recorte}*{avaliacao}.*"
    ):
        if p.suffix.lower() in {".csv", ".tex", ".latex"}:
            candidatos.append(p.with_suffix(""))

    vistos = set()
    unicos = []
    for c in candidatos:
        if str(c) not in vistos:
            vistos.add(str(c))
            unicos.append(c)
    return unicos


def _candidatos_top10_tree(recorte: str, avaliacao: str) -> list[Path]:
    base_exata = _base_tree_article(recorte, avaliacao)
    padroes = [
        f"tabela_top10_importancias_treeClassification_19_ternario_recorte_{recorte}_{avaliacao}",
        f"tabela_top10_importancias_treeClassification_19_profundidade_ternario_recorte_{recorte}_{avaliacao}",
        f"tabela_importancias_treeClassification_19_ternario_recorte_{recorte}_{avaliacao}",
        f"tabela_importancias_treeClassification_19_profundidade_ternario_recorte_{recorte}_{avaliacao}",
    ]

    candidatos: list[Path] = []
    for stem_name in padroes:
        stem = base_exata / stem_name
        for ext in [".csv", ".tex", ".latex"]:
            if stem.with_suffix(ext).exists():
                candidatos.append(stem)
                break

    for p in REPORTS_ARTICLE_DIR.glob(
        f"appendix/apendice_modelagem/**/tabela*importancias*treeClassification*19*ternario*{recorte}*{avaliacao}.*"
    ):
        if p.suffix.lower() in {".csv", ".tex", ".latex"}:
            candidatos.append(p.with_suffix(""))

    vistos = set()
    unicos = []
    for c in candidatos:
        if str(c) not in vistos:
            vistos.add(str(c))
            unicos.append(c)
    return unicos


def gerar_tabela_compacta_tree_existente_cinza(recorte: str, avaliacao: str, output_dir: Path) -> bool:
    """Lê a tabela compacta da árvore já existente e regrava no apêndice em tons de cinza."""
    for stem in _candidatos_tabela_compacta_tree(recorte, avaliacao):
        df = _ler_tabela_csv_ou_tex(stem)
        if df is None or df.empty:
            continue

        for col in list(df.columns):
            if col.strip().lower() == "profundidade":
                valores = df[col].dropna().astype(str).unique()
                if len(valores) <= 1:
                    df = df.drop(columns=[col])

        destino = output_dir / _destino_modelagem(recorte)
        salvar_tabela_cinza(
            df,
            destino,
            f"tabela_compacta_treeClassification_19_profundidade_ternario_recorte_{recorte}_{avaliacao}",
            f"Resumo da árvore de decisão multinomial, recorte {recorte}, avaliação {avaliacao}, profundidade máxima 19.",
            f"tab:tabela_compacta_treeClassification_19_profundidade_ternario_recorte_{recorte}_{avaliacao}",
            notas="Renda, desempenho e interação indicam importâncias normalizadas das variáveis na árvore. A profundidade máxima é 19.",
            gerar_imagem=True,
        )
        return True

    return False


def _normalizar_tabela_importancias(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza tabela de importâncias em Posição, Variável, Importância."""
    if df.empty:
        return df

    cols_lower = {c.lower().strip(): c for c in df.columns}
    if "posição" in cols_lower and "variável" in cols_lower and "importância" in cols_lower:
        out = df[[cols_lower["posição"], cols_lower["variável"], cols_lower["importância"]]].copy()
        out.columns = ["Posição", "Variável", "Importância"]
        return out.head(10)

    var_col = None
    imp_col = None

    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["variavel_original", "variavel", "feature_original", "variável", "feature", "variavel_original_agrupada"]:
            var_col = c
        if lc in ["importancia_normalizada", "importancia", "importance", "importância", "feature_importance"]:
            imp_col = c

    if var_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                var_col = c
                break

    if imp_col is None:
        for c in df.columns:
            serie = pd.to_numeric(df[c], errors="coerce")
            if serie.notna().sum() > 0:
                imp_col = c
                break

    if var_col is None or imp_col is None:
        return pd.DataFrame()

    tmp = df[[var_col, imp_col]].copy()
    tmp[imp_col] = pd.to_numeric(
        tmp[imp_col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    tmp = tmp.dropna(subset=[imp_col])

    if tmp.empty:
        return pd.DataFrame()

    tmp = (
        tmp.groupby(var_col, as_index=False)[imp_col]
        .sum()
        .sort_values(imp_col, ascending=False)
        .head(10)
        .copy()
    )

    tmp.insert(0, "Posição", range(1, len(tmp) + 1))
    tmp["Variável"] = tmp[var_col].map(lambda x: LABEL_VARIAVEIS.get(str(x), str(x)))
    tmp["Importância"] = tmp[imp_col].map(lambda x: _fmt_float_br(x, 4))

    return tmp[["Posição", "Variável", "Importância"]]


def gerar_top10_tree_existente_cinza(recorte: str, avaliacao: str, output_dir: Path) -> bool:
    """Lê tabela top10/importâncias já existente e regrava no apêndice em tons de cinza."""
    for stem in _candidatos_top10_tree(recorte, avaliacao):
        df = _ler_tabela_csv_ou_tex(stem)
        if df is None or df.empty:
            continue

        out = _normalizar_tabela_importancias(df)
        if out.empty:
            continue

        destino = output_dir / _destino_modelagem(recorte)
        salvar_tabela_cinza(
            out,
            destino,
            f"tabela_top10_importancias_treeClassification_19_ternario_recorte_{recorte}_{avaliacao}",
            f"Dez variáveis de maior importância na árvore de decisão multinomial, recorte {recorte}, avaliação {avaliacao}, E5.",
            f"tab:top10_importancias_treeClassification_19_ternario_{recorte}_{avaliacao}",
            notas="Importâncias normalizadas dentro do modelo. Valores maiores indicam maior contribuição para as partições da árvore, sem interpretação causal.",
            gerar_imagem=True,
        )
        return True

    return False


def copiar_tabelas_tree_existentes(recorte: str, avaliacao: str, output_dir: Path, dry_run: bool = False) -> int:
    """Exporta as tabelas da árvore já existentes, reformatadas em tons de cinza.

    Não copia figuras de árvore.
    """
    if dry_run:
        compactas = _candidatos_tabela_compacta_tree(recorte, avaliacao)
        tops = _candidatos_top10_tree(recorte, avaliacao)
        print(f"[DRY] Candidatas compactas árvore {recorte}: {len(compactas)}")
        for p in compactas:
            print(f"      {p}")
        print(f"[DRY] Candidatas top10 árvore {recorte}: {len(tops)}")
        for p in tops:
            print(f"      {p}")
        return 0

    total = 0
    if gerar_tabela_compacta_tree_existente_cinza(recorte, avaliacao, output_dir):
        total += 1
    if gerar_top10_tree_existente_cinza(recorte, avaliacao, output_dir):
        total += 1

    return total


def gerar_modelagem(output_dir: Path, avaliacao: str, dry_run: bool = False) -> None:
    for recorte in ["geral", "medicina"]:
        if not dry_run:
            gerar_efeitos_multinomiais_se_possivel(recorte, avaliacao)

        copiar_efeitos_multinomiais(recorte, avaliacao, output_dir, dry_run=dry_run)

        if dry_run:
            copiar_tabelas_tree_existentes(recorte, avaliacao, output_dir, dry_run=True)
            continue

        gerar_tabela_compacta_logit_ternario(recorte, avaliacao, output_dir)
        gerar_tabela_coeficientes_logit_ternario(recorte, avaliacao, output_dir)

        # Para evitar tabelas antigas/stale, as tabelas da árvore usadas nos apêndices
        # são sempre reconstruídas diretamente dos diagnostics do modelo treinado.
        # Em especial, o top 10 usa somente *_importancias_agregadas.csv; nunca mistura
        # importâncias transformadas e agregadas.
        try:
            gerar_tabela_compacta_tree_19_ternario(recorte, avaliacao, output_dir)
        except Exception as exc:
            print(f"[AVISO] Métricas da árvore 19 não exportadas para recorte={recorte}, avaliação={avaliacao}: {exc}")

        try:
            gerar_top10_importancias_tree_19_ternario(recorte, avaliacao, output_dir)
        except Exception as exc:
            print(f"[AVISO] Importâncias da árvore 19 não exportadas para recorte={recorte}, avaliação={avaliacao}: {exc}")

        # Duas tabelas adicionais apenas para o Apêndice C:
        # árvore binária da etapa contratual, profundidade 10, recorte geral.
        if recorte == "geral":
            gerar_tabelas_tree_10_binario_geral(avaliacao, output_dir)



def escrever_textos_guia(output_dir: Path) -> None:
    textos = {
        "apendices/apendice_a_fluxo_selecao/LEIA_ME.txt": (
            "Apêndice A – Análises suplementares do fluxo de seleção\n\n"
            "Reúne figuras complementares da seção 4.1.\n"
        ),
        "apendices/apendice_b_desfechos_renda_desempenho/LEIA_ME.txt": (
            "Apêndice B – Desfechos segundo renda e desempenho\n\n"
            "Reúne a distribuição das inscrições por faixa de renda e a matriz de lista de espera.\n"
        ),
        "apendices/apendice_c_modelagem_geral/LEIA_ME.txt": (
            "Apêndice C – Evidências complementares de modelagem no recorte geral\n\n"
            "Reúne seis tabelas: especificações do logit multinomial, coeficientes-chave do logit multinomial, "
            "resumo e top 10 importâncias da árvore com três situações, além do resumo e top 10 da árvore restrita à etapa contratual.\n"
        ),
        "apendices/apendice_d_modelagem_medicina/LEIA_ME.txt": (
            "Apêndice D – Evidências complementares de modelagem no recorte Medicina\n\n"
            "Reúne quatro tabelas: especificações do logit multinomial, coeficientes-chave do logit multinomial, "
            "resumo da árvore de decisão e top 10 importâncias da árvore.\n"
        ),
    }

    for rel, conteudo in textos.items():
        path = output_dir / rel
        _ensure_dir(path.parent)
        path.write_text(conteudo, encoding="utf-8")


def gerar_manifesto(output_dir: Path) -> None:
    arquivos = sorted(
        p for p in output_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSOES_EXPORTADAS | {".txt"}
    )

    linhas = ["arquivo,tamanho_bytes"]
    for p in arquivos:
        linhas.append(f'"{p.relative_to(output_dir)}",{p.stat().st_size}')

    (output_dir / "manifesto_exportacao.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"[OK] Manifesto: {output_dir / 'manifesto_exportacao.csv'}")


def run(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    avaliacao: str = "in_sample",
    clean: bool = False,
    dry_run: bool = False,
) -> None:
    output_dir = Path(output_dir).resolve()

    if not REPORTS_ARTICLE_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {REPORTS_ARTICLE_DIR}")

    if clean and not dry_run:
        _limpar_pasta(output_dir)
    else:
        _ensure_dir(output_dir)

    print("=" * 80)
    print("PACOTE FINAL DO ARTIGO")
    print("=" * 80)
    print(f"Origem reports/article: {REPORTS_ARTICLE_DIR}")
    print(f"Origem diagnostics:     {DIAGNOSTICS_MODELING_DIR}")
    print(f"Destino:               {output_dir}")
    print(f"Avaliação modelagem:   {avaliacao}")
    print(f"Dry-run:               {'sim' if dry_run else 'não'}")
    print("-" * 80)

    copiar_elementos_fixos(output_dir, dry_run=dry_run)
    gerar_modelagem(output_dir, avaliacao=avaliacao, dry_run=dry_run)

    if not dry_run:
        escrever_textos_guia(output_dir)
        gerar_manifesto(output_dir)

    print("-" * 80)
    print("[OK] Exportação concluída.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--avaliacao", default="in_sample", choices=["in_sample", "holdout_80_20"])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(output_dir=args.out, avaliacao=args.avaliacao, clean=args.clean, dry_run=args.dry_run)
