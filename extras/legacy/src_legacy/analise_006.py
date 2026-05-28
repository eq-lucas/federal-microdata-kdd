# %%
def gerar_figura_b1_lista_espera():

    from constantes import pasta_data_04_load_database, pasta_data_05_processed

    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from matplotlib import font_manager

    # ==============================================================================
    # 1. CONFIGURAÇÕES
    # ==============================================================================

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", "{:.2f}".format)

    sns.set_theme(style="white", font_scale=1.0)

    fontes_disponiveis = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in fontes_disponiveis:
        fonte_padrao = "Times New Roman"
    elif "Liberation Serif" in fontes_disponiveis:
        fonte_padrao = "Liberation Serif"
    elif "Nimbus Roman" in fontes_disponiveis:
        fonte_padrao = "Nimbus Roman"
    else:
        fonte_padrao = "DejaVu Serif"

    plt.rcParams["font.family"] = fonte_padrao
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 1200

    caminho_final_parquet = pasta_data_05_processed / "analise_b1_lista_espera.parquet"
    pasta_figuras = Path("../reports/figures/apendices").resolve()
    pasta_figuras.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------
    # Tons de cinza com contraste, sem preto muito forte
    # ------------------------------------------------------------------------------
    cmap_cinza_contraste = LinearSegmentedColormap.from_list(
        "cinza_contraste_suave",
        [
            "#f9f9f9",  # quase branco
            "#e3e3e3",  # cinza muito claro
            "#c2c2c2",  # cinza claro
            "#919191",  # cinza médio
            "#6a6a6a",  # cinza escuro moderado
        ],
        N=256
    )

    # Normalização para destacar diferenças sem escurecer demais
    norma_cinza = PowerNorm(
        gamma=0.58,
        vmin=0,
        vmax=100
    )

    # ==============================================================================
    # 2. LOAD + FILTRO
    # ==============================================================================

    print("[*] Carregando base...")
    caminho_base = pasta_data_04_load_database / "inscritos_final_limpo.parquet"
    df = pd.read_parquet(str(caminho_base))

    df["modalidade_fies"] = (
        df["modalidade_fies"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Mesmo recorte usado nas demais matrizes:
    # opção prioritária e Modalidade I
    df = df[
        (df["opcao_curso"] == 1) &
        (df["modalidade_fies"] == "MODALIDADE I")
    ].copy()

    print(f"Total após filtro: {len(df):,}".replace(",", "."))

    # ==============================================================================
    # 3. FAIXAS
    # ==============================================================================

    bins_renda = [-np.inf, 600, 1200, 1800, 2400, 3000, np.inf]

    labels_renda_full = [
        "0–600",
        "601–1.200",
        "1.201–1.800",
        "1.801–2.400",
        "2.401–3.000",
        "> 3.000"
    ]

    labels_renda_plot = ["I", "II", "III", "IV", "V", "VI"]

    df["renda_per_capita"] = pd.to_numeric(df["renda_per_capita"], errors="coerce")
    df = df.dropna(subset=["renda_per_capita"]).copy()

    df["faixa_renda_bruta"] = pd.cut(
        df["renda_per_capita"],
        bins=bins_renda,
        labels=labels_renda_full
    )

    df["media_enem"] = pd.to_numeric(df["media_enem"], errors="coerce")
    df["nota_corte_gp"] = pd.to_numeric(df["nota_corte_gp"], errors="coerce")
    df = df.dropna(subset=["media_enem", "nota_corte_gp"]).copy()

    df["gap"] = df["media_enem"] - df["nota_corte_gp"]

    bins_gap = [-np.inf, -150, -50, 0, 50, 150, np.inf]

    labels_gap = [
        "< -150",
        "[-150, -50]",
        "[-50, 0]",
        "[0, +50]",
        "[+50, +150]",
        "> +150"
    ]

    df["nivel_nota_gap"] = pd.cut(
        df["gap"],
        bins=bins_gap,
        labels=labels_gap
    )

    # ==============================================================================
    # 4. STATUS
    # ==============================================================================

    def normalizar(s):
        return str(s).strip().upper()

    mapa_status = {
        "CONTRATADA": "1. CONTRATADA",
        "INSCRIÇÃO POSTERGADA": "2. INSCRIÇÃO POSTERGADA",
        "PRE-SELECIONADO": "3. PRÉ-SELECIONADO",
        "PRÉ-SELECIONADO": "3. PRÉ-SELECIONADO",
        "NÃO CONTRATADO": "4. NÃO CONTRATADO",
        "NAO CONTRATADO": "4. NÃO CONTRATADO",
        "REJEITADA PELA CPSA": "5. REJEITADA PELA CPSA",
        "OPÇÃO NÃO CONTRATADA": "6. OPÇÃO NÃO CONTRATADA",
        "OPCAO NAO CONTRATADA": "6. OPÇÃO NÃO CONTRATADA",
        "PARTICIPACAO CANCELADA PELO CANDIDATO": "7. PARTICIPACAO CANCELADA",
        "PARTICIPAÇÃO CANCELADA PELO CANDIDATO": "7. PARTICIPACAO CANCELADA",
        "LISTA DE ESPERA": "8. LISTA DE ESPERA"
    }

    df["status_norm"] = df["situacao_fies"].apply(normalizar)
    df["status_final"] = df["status_norm"].map(mapa_status).fillna("9. OUTROS")

    # ==============================================================================
    # 5. CHECK DE INTEGRIDADE
    # ==============================================================================

    print("\n🔎 CHECK DE INTEGRIDADE:")
    print("Sem renda:", df["faixa_renda_bruta"].isna().sum())
    print("Sem nota:", df["nivel_nota_gap"].isna().sum())
    print("Sem status:", df["status_final"].isna().sum())

    # ==============================================================================
    # 6. PREPARAÇÃO DA BASE ANALÍTICA
    # ==============================================================================

    df_counts = (
        df.groupby(
            [
                "faixa_renda_bruta",
                "nivel_nota_gap",
                "status_final"
            ],
            observed=True
        )
        .size()
        .reset_index(name="qtd")
    )

    df_totals = (
        df.groupby(
            [
                "faixa_renda_bruta",
                "nivel_nota_gap"
            ],
            observed=True
        )
        .size()
        .reset_index(name="total_celula")
    )

    df_analise = df_counts.merge(
        df_totals,
        on=["faixa_renda_bruta", "nivel_nota_gap"],
        how="left"
    )

    df_analise["percentual_celula"] = (
        df_analise["qtd"] / df_analise["total_celula"] * 100
    )

    df_analise.to_parquet(str(caminho_final_parquet), index=False)

    # ==============================================================================
    # 7. ISOLA SOMENTE LISTA DE ESPERA
    # ==============================================================================

    df_lista = df_analise[
        df_analise["status_final"] == "8. LISTA DE ESPERA"
    ].copy()

    ordem_nota_plot = labels_gap[::-1]

    if df_lista.empty:
        matriz = pd.DataFrame(
            0.0,
            index=ordem_nota_plot,
            columns=labels_renda_full
        )
    else:
        matriz = df_lista.pivot(
            index="nivel_nota_gap",
            columns="faixa_renda_bruta",
            values="percentual_celula"
        )

        matriz = matriz.reindex(
            index=ordem_nota_plot,
            columns=labels_renda_full
        ).fillna(0)

    # ------------------------------------------------------------------------------
    # Ajusta vmax dinamicamente para evitar que tudo fique muito claro
    # ------------------------------------------------------------------------------
    vmax_real = float(np.nanmax(matriz.to_numpy())) if matriz.size > 0 else 0.0
    vmax_usado = max(10, min(100, np.ceil(vmax_real / 5) * 5))

    norma_cinza = PowerNorm(
        gamma=0.58,
        vmin=0,
        vmax=vmax_usado
    )

    # ==============================================================================
    # 8. PLOT
    # ==============================================================================

    def formatar_anotacao(valor):
        return f"{valor:.1f}".replace(".", ",")

    def ajustar_cor_textos_heatmap(ax, limiar_branco=0.58):
        """
        Deixa o texto branco apenas nas células mais escuras,
        usando proporção do vmax efetivo.
        """
        limiar_valor = vmax_usado * limiar_branco

        for texto in ax.texts:
            try:
                valor = float(texto.get_text().replace(",", "."))
            except ValueError:
                continue

            if valor >= limiar_valor:
                texto.set_color("white")
            else:
                texto.set_color("black")

            texto.set_fontweight("bold")

    matriz_annot = matriz.map(formatar_anotacao)

    # Figura um pouco mais compacta para reduzir ligeiramente as células,
    # mantendo os textos bem grandes.
    fig, ax = plt.subplots(
        figsize=(11.4, 8.8),
        dpi=450
    )

    sns.heatmap(
        matriz,
        annot=matriz_annot,
        fmt="",
        cmap=cmap_cinza_contraste,
        norm=norma_cinza,
        linewidths=0.45,
        linecolor="#bebebe",
        annot_kws={
            "size": 31,
            "weight": "bold"
        },
        cbar=False,
        square=False,
        ax=ax
    )

    ajustar_cor_textos_heatmap(ax, limiar_branco=0.58)

    # Sem título interno
    ax.set_title("")

    ax.set_xlabel(
        "Faixa de renda familiar per capita",
        fontweight="bold",
        fontsize=27,
        color="black",
        labelpad=8
    )

    ax.set_ylabel(
        "Desempenho relativo à nota de corte",
        fontweight="bold",
        fontsize=27,
        color="black",
        labelpad=8
    )

    ax.set_xticklabels(
        labels_renda_plot,
        rotation=0,
        fontsize=25,
        color="black"
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=24,
        color="black"
    )

    ax.tick_params(axis="x", colors="black", pad=1)
    ax.tick_params(axis="y", colors="black", pad=1)

    for lado in ["top", "right", "left", "bottom"]:
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_linewidth(1.0)
        ax.spines[lado].set_color("black")

    plt.subplots_adjust(
        left=0.16,
        right=0.995,
        bottom=0.12,
        top=0.995
    )

    nome_base = re.sub(r"\W+", "_", "figura_b1_lista_espera".lower()).strip("_")

    caminho_png = pasta_figuras / f"{nome_base}.png"
    caminho_tiff = pasta_figuras / f"{nome_base}.tiff"
    caminho_pdf = pasta_figuras / f"{nome_base}.pdf"

    fig.savefig(
        caminho_png,
        dpi=1200,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white"
    )

    fig.savefig(
        caminho_tiff,
        dpi=1200,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white"
    )

    fig.savefig(
        caminho_pdf,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white"
    )

    plt.close(fig)

    print(f"\n✅ Figura B1 exportada para: {caminho_png}")
    print(f"✅ Figura B1 em TIFF exportada para: {caminho_tiff}")
    print(f"✅ Figura B1 em PDF exportada para: {caminho_pdf}")


gerar_figura_b1_lista_espera()
# %%