def eda_gerar_funil_de_selecao_6_etapas_v2():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as ticker
    from matplotlib.patches import Patch
    from matplotlib import font_manager
    from pathlib import Path
    import re

    from src.constantes import pasta_data_05_processed

    # ==============================================================================
    # 0. CONFIGURAÇÕES GERAIS
    # ==============================================================================

    FIG_DPI = 180
    SAVE_DPI = 300

    plt.rcParams["figure.dpi"] = FIG_DPI
    plt.rcParams["savefig.dpi"] = SAVE_DPI

    # Hachura mais fina para não escurecer demais as partes pequenas das barras
    plt.rcParams["hatch.linewidth"] = 0.65

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

    pasta_saida = Path("../analise006_1_1")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # 1. PREPARAÇÃO DOS DADOS
    # ==============================================================================

    path_funil_regiao = pasta_data_05_processed / "funil_por_regiao.parquet"
    df = pd.read_parquet(str(path_funil_regiao))

    if "regiao_ies" in df.columns:
        df = df.rename(columns={"regiao_ies": "regiao"})

    if "regiao" in df.columns:
        df["regiao"] = df["regiao"].fillna("Região não informada")

    df["periodo"] = df["ano"].astype(str) + "." + df["semestre"].astype(str)

    df["nome_cine_area_geral"] = df["nome_cine_area_geral"].fillna(
        "CINE Não Informado (MEC)"
    )

    ordem_areas_cine = [
        "Saúde e bem-estar",
        "Negócios, administração e direito",
        "Engenharia, produção e construção",
        "Ciências sociais, comunicação e informação",
        "Agricultura, silvicultura, pesca e veterinária",
        "Educação",
        "Computação e Tecnologias da Informação e Comunicação (TIC)",
        "Computação e Tecnologias da Informação e Comunicação",
        "Serviços",
        "Artes e humanidades",
        "Ciências naturais, matemática e estatística",
    ]

    areas_presentes = df["nome_cine_area_geral"].unique().tolist()
    areas = [area for area in ordem_areas_cine if area in areas_presentes]

    areas_restantes = [area for area in areas_presentes if area not in areas]
    areas.extend(sorted(areas_restantes))

    periodos = sorted(df["periodo"].unique())
    regioes = sorted(df["regiao"].unique())

    # ==============================================================================
    # 2. COLUNAS DO FUNIL
    # ==============================================================================

    colunas_inscritos = [
        "vagas_fies",
        "Inscritos_Geral",
        "inscritos_com_nota_suficiente",
        "vagas_ocupadas",
    ]

    colunas_candidatos = [
        "vagas_fies",
        "Candidatos_Unicos_Geral",
        "candidatos_unicos_com_nota_suficiente",
        "vagas_ocupadas",
    ]

    nomes_curto = {
        "vagas_fies": "I",
        "Inscritos_Geral": "II",
        "inscritos_com_nota_suficiente": "III",
        "Candidatos_Unicos_Geral": "II",
        "candidatos_unicos_com_nota_suficiente": "III",
        "vagas_ocupadas": "IV",
    }

    # ==============================================================================
    # 3. FUNÇÕES AUXILIARES
    # ==============================================================================

    def limpar_nome_arquivo(texto):
        texto = str(texto).lower()

        substituicoes = {
            "ç": "c",
            "ã": "a",
            "á": "a",
            "à": "a",
            "â": "a",
            "ä": "a",
            "é": "e",
            "ê": "e",
            "è": "e",
            "ë": "e",
            "í": "i",
            "ì": "i",
            "î": "i",
            "ï": "i",
            "ó": "o",
            "õ": "o",
            "ô": "o",
            "ò": "o",
            "ö": "o",
            "ú": "u",
            "ù": "u",
            "û": "u",
            "ü": "u",
        }

        for original, novo in substituicoes.items():
            texto = texto.replace(original, novo)

        texto = re.sub(r"[^a-z0-9]+", "_", texto)
        texto = re.sub(r"_+", "_", texto)

        return texto.strip("_")

    def estilos_monocromaticos(categorias_list):
        tons_cinza = [
            "0.22",
            "0.82",
            "0.38",
            "0.68",
            "0.52",
            "0.90",
            "0.30",
            "0.76",
            "0.46",
            "0.94",
            "0.60",
            "0.86",
        ]

        hachuras = [
            "",
            "/",
            "\\",
            "x",
            "-",
            ".",
            "|",
            "+",
            "o",
            "*",
            "//",
            "\\\\",
        ]

        estilos = {}

        for i, categoria in enumerate(categorias_list):
            estilos[categoria] = {
                "cor": tons_cinza[i % len(tons_cinza)],
                "hatch": hachuras[i % len(hachuras)],
            }

        return estilos

    def plot_funil_seguro(
        df_data,
        periodos_list,
        categorias_list,
        col_categoria,
        cols_metricas,
        nome_arquivo,
        titulo_legenda=None,
        figsize=(24, 32),
        y_step=None,
        salvar=True,
        mostrar=True,
    ):

        # --------------------------------------------------------------------------
        # 1. Agrupamento
        # --------------------------------------------------------------------------

        if col_categoria is not None:
            df_agg = (
                df_data
                .groupby(["periodo", col_categoria])[cols_metricas]
                .sum()
                .reset_index()
            )

            categorias_list = list(categorias_list)

            mux = pd.MultiIndex.from_product(
                [periodos_list, categorias_list],
                names=["periodo", col_categoria],
            )

            df_plot = (
                df_agg
                .set_index(["periodo", col_categoria])
                .reindex(mux, fill_value=0)
                .reset_index()
            )

        else:
            df_plot = (
                df_data
                .groupby(["periodo"])[cols_metricas]
                .sum()
                .reset_index()
            )

            df_plot["dummy"] = "Total"
            categorias_list = ["Total"]
            col_categoria = "dummy"

        tem_legenda = col_categoria != "dummy"

        # --------------------------------------------------------------------------
        # 2. Figura
        # --------------------------------------------------------------------------

        if tem_legenda:
            # Altura menor para a legenda. O gráfico ganha mais espaço vertical.
            if col_categoria == "nome_cine_area_geral":
                altura_legenda = 0.16
            elif col_categoria == "regiao":
                altura_legenda = 0.09
            else:
                altura_legenda = 0.11

            fig = plt.figure(figsize=figsize, dpi=FIG_DPI)

            gs = fig.add_gridspec(
                nrows=2,
                ncols=1,
                height_ratios=[altura_legenda, 1.0 - altura_legenda],
                hspace=0.0,
            )

            ax_legenda = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            ax_legenda.axis("off")

        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=FIG_DPI)
            ax_legenda = None

        # --------------------------------------------------------------------------
        # 3. Barras empilhadas
        # --------------------------------------------------------------------------

        x = np.arange(len(cols_metricas))

        largura_barra = 0.155
        half_span = largura_barra * (len(periodos_list) - 1) / 2.0
        deslocamentos = np.linspace(-half_span, half_span, len(periodos_list))

        estilos = estilos_monocromaticos(categorias_list)

        for desloc, periodo in zip(deslocamentos, periodos_list):
            df_periodo = df_plot[df_plot["periodo"] == periodo]
            base = np.zeros(len(cols_metricas))

            for cat in categorias_list:
                valores_originais = (
                    df_periodo[df_periodo[col_categoria] == cat][cols_metricas]
                    .values[0]
                )

                valores_scaled = np.array(valores_originais) / 1000.0

                ax.bar(
                    x + desloc,
                    valores_scaled,
                    width=largura_barra,
                    bottom=base,
                    color=estilos[cat]["cor"],
                    edgecolor="black",
                    linewidth=0.65,
                    hatch=estilos[cat]["hatch"],
                    zorder=3,
                )

                base += valores_scaled

        # --------------------------------------------------------------------------
        # 4. Eixo X
        # --------------------------------------------------------------------------

        posicoes_ticks = []
        labels_ticks = []

        for i in range(len(cols_metricas)):
            for desloc, periodo in zip(deslocamentos, periodos_list):
                posicoes_ticks.append(i + desloc)
                ano, sem = periodo.split(".")
                labels_ticks.append(f"'{ano[-2:]}.{sem}")

        ax.set_xticks(posicoes_ticks)

        ax.set_xticklabels(
            labels_ticks,
            rotation=90,
            ha="center",
            fontsize=26,
            fontweight="normal",
            color="black",
        )

        ax.tick_params(axis="x", colors="black", pad=4)
        ax.set_xlim(-0.5, len(cols_metricas) - 0.5)

        for i in range(1, len(cols_metricas)):
            ax.axvline(
                x=i - 0.5,
                color="0.70",
                linestyle="--",
                linewidth=1.25,
                zorder=1,
            )

        for i, col in enumerate(cols_metricas):
            ax.annotate(
                nomes_curto.get(col, col),
                xy=(i, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -78),
                textcoords="offset points",
                ha="center",
                va="top",
                fontweight="bold",
                fontsize=34,
                color="black",
                annotation_clip=False,
            )

        # --------------------------------------------------------------------------
        # 5. Eixo Y
        # --------------------------------------------------------------------------

        ax.set_ylabel(
            "Quantidade (mil)",
            fontsize=32,
            fontweight="bold",
            color="black",
            labelpad=14,
        )

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda val, pos: f"{val:,.0f}".replace(",", "."))
        )

        ax.tick_params(axis="y", labelsize=24, colors="black")

        if y_step is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.45,
            color="0.65",
            zorder=0,
        )

        ax.set_ylim(bottom=0)

        # Mantém todas as bordas do gráfico.
        for lado in ["top", "right", "left", "bottom"]:
            ax.spines[lado].set_visible(True)
            ax.spines[lado].set_linewidth(1.0)
            ax.spines[lado].set_color("black")

        # --------------------------------------------------------------------------
        # 6. Legenda com borda visível e pouco padding
        # --------------------------------------------------------------------------

        if tem_legenda:
            handles_custom = [
                Patch(
                    facecolor=estilos[cat]["cor"],
                    edgecolor="black",
                    linewidth=1.05,
                    hatch=estilos[cat]["hatch"],
                    label=cat,
                )
                for cat in categorias_list
            ]

            if col_categoria == "nome_cine_area_geral":
                num_colunas = 2
                legenda_fontsize = 25
                handlelength = 4.0
                handleheight = 1.45
                columnspacing = 2.0
                labelspacing = 0.45
                handletextpad = 0.75
                borderpad = 0.25

            elif col_categoria == "regiao":
                num_colunas = len(categorias_list)
                legenda_fontsize = 27
                handlelength = 3.8
                handleheight = 1.40
                columnspacing = 1.6
                labelspacing = 0.35
                handletextpad = 0.75
                borderpad = 0.20

            else:
                num_colunas = min(4, len(categorias_list))
                legenda_fontsize = 24
                handlelength = 3.8
                handleheight = 1.35
                columnspacing = 1.6
                labelspacing = 0.35
                handletextpad = 0.75
                borderpad = 0.20

            leg = ax_legenda.legend(
                handles=handles_custom,
                loc="center",
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                mode="expand",
                ncol=num_colunas,
                frameon=True,
                fancybox=False,
                edgecolor="black",
                facecolor="white",
                framealpha=1.0,
                fontsize=legenda_fontsize,
                handlelength=handlelength,
                handleheight=handleheight,
                handletextpad=handletextpad,
                columnspacing=columnspacing,
                labelspacing=labelspacing,
                borderaxespad=0.0,
                borderpad=borderpad,
            )

            leg.get_frame().set_linewidth(1.0)

        # --------------------------------------------------------------------------
        # 7. Ajuste da imagem inteira
        # --------------------------------------------------------------------------

        if tem_legenda:
            fig.subplots_adjust(
                left=0.055,
                right=0.995,
                top=0.992,
                bottom=0.080,
            )
        else:
            fig.subplots_adjust(
                left=0.075,
                right=0.995,
                top=0.985,
                bottom=0.120,
            )

        # --------------------------------------------------------------------------
        # 8. Salvar e/ou mostrar
        # --------------------------------------------------------------------------

        if salvar:
            nome_limpo = limpar_nome_arquivo(nome_arquivo)

            caminho_png = pasta_saida / f"{nome_limpo}.png"
            caminho_tiff = pasta_saida / f"{nome_limpo}.tiff"

            fig.savefig(
                caminho_png,
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=0.03,
                facecolor="white",
            )

            fig.savefig(
                caminho_tiff,
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=0.03,
                facecolor="white",
            )

        if mostrar:
            plt.show()
        else:
            plt.close(fig)

    # ==============================================================================
    # 4. EXECUÇÃO DO PIPELINE
    # ==============================================================================

    plot_funil_seguro(
        df_data=df,
        periodos_list=periodos,
        categorias_list=areas,
        col_categoria="nome_cine_area_geral",
        cols_metricas=colunas_candidatos,
        nome_arquivo="figura_1a_funil_fies_candidatos_unicos_area_cine",
        titulo_legenda="Área CINE",
        figsize=(24, 30),
        y_step=100,
        salvar=True,
        mostrar=True,
    )

    plot_funil_seguro(
        df_data=df,
        periodos_list=periodos,
        categorias_list=areas,
        col_categoria="nome_cine_area_geral",
        cols_metricas=colunas_inscritos,
        nome_arquivo="figura_1_funil_fies_inscritos_area_cine",
        titulo_legenda="Área CINE",
        figsize=(24, 30),
        y_step=100,
        salvar=True,
        mostrar=True,
    )

    plot_funil_seguro(
        df_data=df,
        periodos_list=periodos,
        categorias_list=None,
        col_categoria=None,
        cols_metricas=colunas_candidatos,
        nome_arquivo="funil_fies_candidatos_unicos_global_consolidado",
        titulo_legenda=None,
        figsize=(20, 22),
        y_step=100,
        salvar=True,
        mostrar=True,
    )

    plot_funil_seguro(
        df_data=df,
        periodos_list=periodos,
        categorias_list=None,
        col_categoria=None,
        cols_metricas=colunas_inscritos,
        nome_arquivo="funil_fies_inscritos_global_consolidado",
        titulo_legenda=None,
        figsize=(20, 22),
        y_step=100,
        salvar=True,
        mostrar=True,
    )

    for area in areas:
        df_area = df[df["nome_cine_area_geral"] == area]
        nome_area = limpar_nome_arquivo(area)

        plot_funil_seguro(
            df_data=df_area,
            periodos_list=periodos,
            categorias_list=regioes,
            col_categoria="regiao",
            cols_metricas=colunas_candidatos,
            nome_arquivo=f"funil_fies_candidatos_unicos_regiao_{nome_area}",
            titulo_legenda="Região",
            figsize=(22, 24),
            y_step=None,
            salvar=True,
            mostrar=True,
        )

        plot_funil_seguro(
            df_data=df_area,
            periodos_list=periodos,
            categorias_list=regioes,
            col_categoria="regiao",
            cols_metricas=colunas_inscritos,
            nome_arquivo=f"funil_fies_inscritos_regiao_{nome_area}",
            titulo_legenda="Região",
            figsize=(22, 24),
            y_step=None,
            salvar=True,
            mostrar=True,
        )

    plot_funil_seguro(
        df_data=df,
        periodos_list=periodos,
        categorias_list=regioes,
        col_categoria="regiao",
        cols_metricas=colunas_candidatos,
        nome_arquivo="funil_fies_candidatos_unicos_regiao_total",
        titulo_legenda="Região",
        figsize=(22, 24),
        y_step=100,
        salvar=True,
        mostrar=True,
    )

    plot_funil_seguro(
        df_data=df,
        periodos_list=periodos,
        categorias_list=regioes,
        col_categoria="regiao",
        cols_metricas=colunas_inscritos,
        nome_arquivo="funil_fies_inscritos_regiao_total",
        titulo_legenda="Região",
        figsize=(22, 24),
        y_step=100,
        salvar=True,
        mostrar=True,
    )