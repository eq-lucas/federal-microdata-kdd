#!/usr/bin/env python3

import sys


def tem_flag(nome: str) -> bool:
    return nome in sys.argv


def valor_flag(nome: str, padrao=None):
    if nome not in sys.argv:
        return padrao
    indice = sys.argv.index(nome)
    if indice + 1 >= len(sys.argv):
        return padrao
    proximo = sys.argv[indice + 1]
    if proximo.startswith("--"):
        return padrao
    return proximo


def obter_recorte(padrao: str = "geral") -> str:
    recorte = valor_flag("--recorte", padrao)
    if recorte == "general":
        recorte = "geral"
    if recorte not in ["geral", "medicina"]:
        print(f"Recorte inválido: {recorte}")
        print("Use: --recorte geral ou --recorte medicina")
        sys.exit(1)
    return recorte


def obter_avaliacao(padrao: str = "all") -> str:
    avaliacao = valor_flag("--avaliacao", padrao)
    if avaliacao not in ["in_sample", "holdout_80_20", "all"]:
        print(f"Avaliação inválida: {avaliacao}")
        print("Use: --avaliacao in_sample, --avaliacao holdout_80_20 ou --avaliacao all")
        sys.exit(1)
    return avaliacao


def obter_experimento(padrao: str = "E5") -> str:
    experimento = str(valor_flag("--experimento", padrao)).upper()
    if experimento not in ["E1", "E2", "E3", "E4", "E5"]:
        print(f"Experimento inválido: {experimento}")
        print("Use: --experimento E1, E2, E3, E4 ou E5")
        sys.exit(1)
    return experimento


def obter_target(padrao: str = "binario") -> str:
    target = valor_flag("--target", padrao)
    if target not in ["binario", "ternario", "all"]:
        print(f"Target inválido: {target}")
        print("Use: --target binario, --target ternario ou --target all")
        sys.exit(1)
    return target


def obter_profundidade(padrao: int = 10) -> int:
    valor = valor_flag("--profundidade", str(padrao))
    try:
        profundidade = int(valor)
    except Exception:
        print(f"Profundidade inválida: {valor}")
        sys.exit(1)
    if profundidade not in [10, 14, 19]:
        print("Profundidade inválida. Use: --profundidade 10, 14 ou 19")
        sys.exit(1)
    return profundidade


def ajuda():
    print("""
================================================================================
 FEDERAL MICRODATA KDD - FIES
================================================================================

Uso:
  python3 main.py [comando] [etapa] [opções]

Preparação:
  init
  check

Fluxo principal:
  reproduce all                        refaz dados, ABTs, produtos finais e exporta article/
  reproduce all --refit                também retreina os modelos antes de exportar article/
  export article                       organiza o pacote final completo em article/

Dados:
  pipeline all                         executa staging + transformações + curadoria
  analysis all                         gera datasets analíticos
  article all                          gera produtos descritivos do artigo

Modelagem:
  modeling logit --force               treina regressões logísticas
  modeling tree --force                treina árvores de decisão padrão
  modeling tree-depth --force          treina árvores de decisão com profundidades 10, 14 e 19
  modeling tree-contratual --force     treina árvore binária da etapa contratual, profundidade 10
  modeling all --force                 treina todos os modelos
  article modelagem                    gera tabelas e figuras dos modelos treinados

ABTs:
  abt binaria                          gera ABT binária geral
  abt ternaria --recorte geral         gera ABT ternária geral
  abt ternaria --recorte medicina      gera ABT ternária Medicina

Comandos individuais:
  transform tipos-fies|unificacao-fies|mestre-inep|cine|modalidade|all
  article fluxo|taxas-conversao|tabelas-distribuicao|matrizes|financiamento
  article logit-binario|logit-ternario|apendice-logit|apendice-logit-ternario
  article efeitos-multinomiais       gera efeitos por probabilidades previstas do logit ternário
  article treeClassification|apendice-treeClassification
  article treeClassification-10|treeClassification-14|treeClassification-19

Opções:
  --force
  --refit
  --target binario|ternario|all
  --recorte geral|medicina|all
  --avaliacao in_sample|holdout_80_20|all
  --experimento E1|E2|E3|E4|E5
  --profundidade 10|14|19
  --dry-run
  --clean
================================================================================
""")


def executar_transform(etapa):
    from src.pipeline.transform import limpeza_tipos_fies, unificacao_fies, mestre_inep, cruzamento_cine, modalidade

    if etapa == "tipos-fies":
        limpeza_tipos_fies.run()
    elif etapa == "unificacao-fies":
        unificacao_fies.run()
    elif etapa == "mestre-inep":
        mestre_inep.run()
    elif etapa == "cine":
        cruzamento_cine.run()
    elif etapa == "modalidade":
        modalidade.run()
    elif etapa == "all":
        limpeza_tipos_fies.run()
        unificacao_fies.run()
        mestre_inep.run()
        cruzamento_cine.run()
        modalidade.run()
    else:
        print(f"Transformação não reconhecida: {etapa}")
        ajuda()


def executar_pipeline(etapa):
    from src.pipeline import staging, curate

    if etapa == "staging":
        staging.run()
    elif etapa == "transform":
        executar_transform("all")
    elif etapa == "curate":
        curate.run()
    elif etapa == "all":
        staging.run()
        executar_transform("all")
        curate.run()
    else:
        print(f"Etapa de pipeline não reconhecida: {etapa}")
        ajuda()


def executar_analysis(etapa):
    from src.analysis import dataset_candidatos_unicos, dataset_funil_fluxo, dataset_taxas

    if etapa == "candidatos-unicos":
        dataset_candidatos_unicos.run()
    elif etapa == "funil":
        dataset_funil_fluxo.run()
    elif etapa == "taxas":
        dataset_taxas.run()
    elif etapa == "all":
        dataset_candidatos_unicos.run()
        dataset_funil_fluxo.run()
        dataset_taxas.run()
    else:
        print(f"Análise não reconhecida: {etapa}")
        ajuda()


def executar_abt(etapa):
    if etapa == "binaria":
        from src.abt import build_abt_binaria
        recorte = obter_recorte("geral")
        build_abt_binaria.run(recorte=recorte)
    elif etapa == "ternaria":
        from src.abt import build_abt_ternaria
        recorte = obter_recorte("geral")
        build_abt_ternaria.run(recorte=recorte)
    else:
        print(f"Etapa de ABT não reconhecida: {etapa}")
        ajuda()


def normalizar_alvo_modelagem(alvo: str) -> list[str]:
    mapa = {"general": "geral", "geral": "geral", "medicina": "medicina"}
    if alvo == "all":
        return ["geral", "medicina"]
    if alvo in mapa:
        return [mapa[alvo]]
    print(f"Alvo de modelagem não reconhecido: {alvo}")
    ajuda()
    sys.exit(1)


def executar_modelagem(acao, alvo="general"):
    force = tem_flag("--force")
    recortes = normalizar_alvo_modelagem(alvo)

    if acao == "all":
        executar_abts_modelagem()
        treinar_todos_modelos(force=force)
        return

    if acao in ["logit", "logit-all"]:
        executar_abts_modelagem()
        treinar_logit_completo(force=force)
        return

    if acao in ["tree", "tree-all"]:
        executar_abts_modelagem()
        treinar_arvore_padrao_completa(force=force)
        return

    if acao in ["tree-depth", "tree-depth-all"]:
        executar_abts_modelagem()
        treinar_arvores_profundidade(force=force)
        return

    if acao in ["tree-contratual", "tree-contract", "tree-binario-contratual"]:
        executar_abts_modelagem()
        treinar_arvore_contratual_binaria(force=force)
        return

    if acao in ["fit", "fit-in-sample", "fit-holdout"]:
        from src.modeling import fit_logit_binario_in_sample, fit_logit_binario_holdout_80_20
        for recorte in recortes:
            if acao in ["fit", "fit-in-sample"]:
                fit_logit_binario_in_sample.run(recorte=recorte, force=force)
            if acao in ["fit", "fit-holdout"]:
                fit_logit_binario_holdout_80_20.run(recorte=recorte, force=force)

    elif acao in ["fit-ternario", "fit-ternario-in-sample", "fit-ternario-holdout"]:
        from src.modeling import fit_logit_ternario_in_sample, fit_logit_ternario_holdout_80_20
        for recorte in recortes:
            if acao in ["fit-ternario", "fit-ternario-in-sample"]:
                fit_logit_ternario_in_sample.run(recorte=recorte, force=force)
            if acao in ["fit-ternario", "fit-ternario-holdout"]:
                fit_logit_ternario_holdout_80_20.run(recorte=recorte, force=force)

    elif acao in ["treeClassification-fit-binario", "treeClassification-fit-binario-in-sample", "treeClassification-fit-binario-holdout"]:
        from src.modeling import treeClassification_binario_in_sample, treeClassification_binario_holdout_80_20
        for recorte in recortes:
            if acao in ["treeClassification-fit-binario", "treeClassification-fit-binario-in-sample"]:
                treeClassification_binario_in_sample.run(recorte=recorte, force=force)
            if acao in ["treeClassification-fit-binario", "treeClassification-fit-binario-holdout"]:
                treeClassification_binario_holdout_80_20.run(recorte=recorte, force=force)

    elif acao in ["treeClassification-fit-ternario", "treeClassification-fit-ternario-in-sample", "treeClassification-fit-ternario-holdout"]:
        from src.modeling import treeClassification_ternario_in_sample, treeClassification_ternario_holdout_80_20
        for recorte in recortes:
            if acao in ["treeClassification-fit-ternario", "treeClassification-fit-ternario-in-sample"]:
                treeClassification_ternario_in_sample.run(recorte=recorte, force=force)
            if acao in ["treeClassification-fit-ternario", "treeClassification-fit-ternario-holdout"]:
                treeClassification_ternario_holdout_80_20.run(recorte=recorte, force=force)

    elif acao.startswith("treeClassification-profundidade-fit-") or acao.startswith("treeClassification-"):
        executar_modelagem_tree_profundidade(acao, recortes, force)

    else:
        print(f"Ação de modelagem não reconhecida: {acao}")
        ajuda()


def profundidade_da_acao(acao: str) -> int:
    if acao.startswith("treeClassification-10-"):
        return 10
    if acao.startswith("treeClassification-14-"):
        return 14
    if acao.startswith("treeClassification-19-"):
        return 19
    return obter_profundidade(10)


def target_da_acao_tree_profundidade(acao: str) -> str:
    if "fit-binario" in acao:
        return "binario"
    if "fit-ternario" in acao:
        return "ternario"
    print(f"Ação de árvore por profundidade inválida: {acao}")
    sys.exit(1)


def executar_modelagem_tree_profundidade(acao: str, recortes: list[str], force: bool):
    from src.modeling.treeClassification_profundidade_utils import run_modelagem

    profundidade = profundidade_da_acao(acao)
    target = target_da_acao_tree_profundidade(acao)

    if "in-sample" in acao:
        avaliacoes = ["in_sample"]
    elif "holdout" in acao:
        avaliacoes = ["holdout_80_20"]
    else:
        avaliacoes = ["in_sample", "holdout_80_20"]

    for recorte in recortes:
        for avaliacao in avaliacoes:
            run_modelagem(target=target, recorte=recorte, avaliacao=avaliacao, force=force, profundidade=profundidade)


def executar_artigo(secao):
    if secao == "fluxo":
        from src.article import fluxo_selecao
        fluxo_selecao.run()
    elif secao in ["taxas-conversao", "taxas_conversao"]:
        from src.article import taxas_conversao
        taxas_conversao.run()
    elif secao in ["tabelas-distribuicao", "tabelas_distribuicao"]:
        from src.article import tabelas_distribuicao
        tabelas_distribuicao.run()
    elif secao == "matrizes":
        from src.article import matrizes_renda_desempenho
        matrizes_renda_desempenho.run()
    elif secao == "financiamento":
        from src.article import financiamento_coparticipacao
        financiamento_coparticipacao.run()

    elif secao in ["logit-binario", "logit_binario"]:
        from src.article import logit_binario
        avaliacao = obter_avaliacao("in_sample")
        if avaliacao == "all":
            print("Para article logit-binario, use --avaliacao in_sample ou holdout_80_20.")
            sys.exit(1)
        logit_binario.run(recorte=obter_recorte("geral"), avaliacao=avaliacao, experimento=obter_experimento("E5"))

    elif secao in ["logit-ternario", "logit_ternario"]:
        from src.article import logit_ternario
        avaliacao = obter_avaliacao("in_sample")
        if avaliacao == "all":
            print("Para article logit-ternario, use --avaliacao in_sample ou holdout_80_20.")
            sys.exit(1)
        logit_ternario.run(recorte=obter_recorte("geral"), avaliacao=avaliacao, experimento=obter_experimento("E5"))

    elif secao in ["apendice-logit", "apendice_logit"]:
        from src.article import apendice_logit_binario
        apendice_logit_binario.run(recorte=obter_recorte("geral"), avaliacao=obter_avaliacao("all"), experimento=obter_experimento("E5"))

    elif secao in ["apendice-logit-ternario", "apendice_logit_ternario"]:
        from src.article import apendice_logit_ternario
        apendice_logit_ternario.run(recorte=obter_recorte("geral"), avaliacao=obter_avaliacao("all"), experimento=obter_experimento("E5"))

    elif secao in ["treeClassification"]:
        from src.article import treeClassification
        avaliacao = obter_avaliacao("in_sample")
        if avaliacao == "all":
            print("Para article treeClassification, use --avaliacao in_sample ou holdout_80_20.")
            sys.exit(1)
        treeClassification.run(target=obter_target("binario"), recorte=obter_recorte("geral"), avaliacao=avaliacao, experimento=obter_experimento("E5"))

    elif secao in ["apendice-treeClassification"]:
        from src.article import apendice_treeClassification
        apendice_treeClassification.run(target=obter_target("binario"), recorte=obter_recorte("geral"), avaliacao=obter_avaliacao("all"), experimento=obter_experimento("E5"))

    elif secao in ["treeClassification-profundidade", "treeClassification-10", "treeClassification-14", "treeClassification-19"]:
        from src.article import treeClassification_profundidade
        if secao == "treeClassification-10":
            profundidade = 10
        elif secao == "treeClassification-14":
            profundidade = 14
        elif secao == "treeClassification-19":
            profundidade = 19
        else:
            profundidade = obter_profundidade(10)
        avaliacao = obter_avaliacao("in_sample")
        if avaliacao == "all":
            print("Para article treeClassification-profundidade, use --avaliacao in_sample ou holdout_80_20.")
            sys.exit(1)
        treeClassification_profundidade.run(
            target=obter_target("binario"),
            recorte=obter_recorte("geral"),
            avaliacao=avaliacao,
            experimento=obter_experimento("E5"),
            profundidade=profundidade,
        )

    elif secao in ["apendice-treeClassification-profundidade", "apendice-treeClassification-10", "apendice-treeClassification-14", "apendice-treeClassification-19"]:
        from src.article import apendice_treeClassification_profundidade
        if secao == "apendice-treeClassification-10":
            profundidade = 10
        elif secao == "apendice-treeClassification-14":
            profundidade = 14
        elif secao == "apendice-treeClassification-19":
            profundidade = 19
        else:
            profundidade = obter_profundidade(10)
        apendice_treeClassification_profundidade.run(
            target=obter_target("binario"),
            recorte=obter_recorte("geral"),
            avaliacao=obter_avaliacao("all"),
            experimento=obter_experimento("E5"),
            profundidade=profundidade,
        )


    elif secao in ["efeitos-multinomiais", "efeitos_multinomiais", "efeitos-ternario", "efeitos_ternario"]:
        from src.article import efeitos_multinomiais_ternario
        avaliacao = obter_avaliacao("in_sample")
        if avaliacao == "all":
            print("Para article efeitos-multinomiais, use --avaliacao in_sample ou holdout_80_20.")
            sys.exit(1)
        efeitos_multinomiais_ternario.run(
            recorte=obter_recorte("geral"),
            avaliacao=avaliacao,
            experimento=obter_experimento("E5"),
        )

    elif secao in ["modelagem", "modelos"]:
        gerar_artigos_modelagem()

    elif secao == "all":
        executar_artigos_descritivos()
    else:
        print(f"Seção do artigo não reconhecida: {secao}")
        ajuda()


def executar_diagnosticos():
    from src.constants import LOGS_DIR, DIAGNOSTICS_DIR
    print("Logs:")
    print(f"  {LOGS_DIR}")
    print()
    print("Diagnósticos:")
    print(f"  {DIAGNOSTICS_DIR}")


def executar_artigos_descritivos():
    from src.article import (
        fluxo_selecao,
        taxas_conversao,
        tabelas_distribuicao,
        matrizes_renda_desempenho,
        financiamento_coparticipacao,
    )

    fluxo_selecao.run()
    taxas_conversao.run()
    tabelas_distribuicao.run()
    matrizes_renda_desempenho.run()
    financiamento_coparticipacao.run()


def executar_abts_modelagem():
    from src.abt import build_abt_binaria, build_abt_ternaria

    print("=" * 80)
    print("ABTS DE MODELAGEM")
    print("=" * 80)

    build_abt_binaria.run(recorte="geral")
    build_abt_ternaria.run(recorte="geral")
    build_abt_ternaria.run(recorte="medicina")


def treinar_logit_completo(force: bool):
    from src.modeling import (
        fit_logit_binario_in_sample,
        fit_logit_binario_holdout_80_20,
        fit_logit_ternario_in_sample,
        fit_logit_ternario_holdout_80_20,
    )

    print("=" * 80)
    print("TREINO | REGRESSÃO LOGÍSTICA")
    print("=" * 80)

    fit_logit_binario_in_sample.run(recorte="geral", force=force)
    fit_logit_binario_holdout_80_20.run(recorte="geral", force=force)

    for recorte in ["geral", "medicina"]:
        fit_logit_ternario_in_sample.run(recorte=recorte, force=force)
        fit_logit_ternario_holdout_80_20.run(recorte=recorte, force=force)


def treinar_arvore_padrao_completa(force: bool):
    from src.modeling import (
        treeClassification_binario_in_sample,
        treeClassification_binario_holdout_80_20,
        treeClassification_ternario_in_sample,
        treeClassification_ternario_holdout_80_20,
    )

    print("=" * 80)
    print("TREINO | ÁRVORES DE DECISÃO | PROFUNDIDADE 6")
    print("=" * 80)

    treeClassification_binario_in_sample.run(recorte="geral", force=force)
    treeClassification_binario_holdout_80_20.run(recorte="geral", force=force)

    for recorte in ["geral", "medicina"]:
        treeClassification_ternario_in_sample.run(recorte=recorte, force=force)
        treeClassification_ternario_holdout_80_20.run(recorte=recorte, force=force)


def treinar_arvores_profundidade(force: bool):
    from src.modeling.treeClassification_profundidade_utils import run_modelagem

    print("=" * 80)
    print("TREINO | ÁRVORES DE DECISÃO | PROFUNDIDADES 10, 14 E 19")
    print("=" * 80)

    for profundidade in [10, 14, 19]:
        print("-" * 80)
        print(f"PROFUNDIDADE {profundidade}")
        print("-" * 80)

        for avaliacao in ["in_sample", "holdout_80_20"]:
            run_modelagem(
                target="binario",
                recorte="geral",
                avaliacao=avaliacao,
                force=force,
                profundidade=profundidade,
            )

        for recorte in ["geral", "medicina"]:
            for avaliacao in ["in_sample", "holdout_80_20"]:
                run_modelagem(
                    target="ternario",
                    recorte=recorte,
                    avaliacao=avaliacao,
                    force=force,
                    profundidade=profundidade,
                )


def treinar_arvore_contratual_binaria(force: bool):
    from src.modeling.treeClassification_profundidade_utils import run_modelagem

    print("=" * 80)
    print("TREINO | ÁRVORE DE DECISÃO | ETAPA CONTRATUAL | PROFUNDIDADE 10")
    print("=" * 80)

    avaliacao = obter_avaliacao("in_sample")
    avaliacoes = ["in_sample", "holdout_80_20"] if avaliacao == "all" else [avaliacao]

    for av in avaliacoes:
        run_modelagem(
            target="binario",
            recorte="geral",
            avaliacao=av,
            force=force,
            profundidade=10,
        )


def treinar_todos_modelos(force: bool):
    treinar_logit_completo(force=force)
    treinar_arvore_padrao_completa(force=force)
    treinar_arvores_profundidade(force=force)


def gerar_artigos_logit():
    from src.article import (
        logit_binario,
        apendice_logit_binario,
        logit_ternario,
        apendice_logit_ternario,
    )

    print("=" * 80)
    print("PRODUTOS | REGRESSÃO LOGÍSTICA")
    print("=" * 80)

    apendice_logit_binario.run(recorte="geral", avaliacao="all", experimento="E5")
    for avaliacao in ["in_sample", "holdout_80_20"]:
        logit_binario.run(recorte="geral", avaliacao=avaliacao, experimento="E5")

    for recorte in ["geral", "medicina"]:
        apendice_logit_ternario.run(recorte=recorte, avaliacao="all", experimento="E5")
        for avaliacao in ["in_sample", "holdout_80_20"]:
            logit_ternario.run(recorte=recorte, avaliacao=avaliacao, experimento="E5")


def gerar_artigos_arvore_padrao():
    from src.article import treeClassification, apendice_treeClassification

    print("=" * 80)
    print("PRODUTOS | ÁRVORES DE DECISÃO | PROFUNDIDADE 6")
    print("=" * 80)

    apendice_treeClassification.run(target="binario", recorte="geral", avaliacao="all", experimento="E5")
    for avaliacao in ["in_sample", "holdout_80_20"]:
        treeClassification.run(target="binario", recorte="geral", avaliacao=avaliacao, experimento="E5")

    for recorte in ["geral", "medicina"]:
        apendice_treeClassification.run(target="ternario", recorte=recorte, avaliacao="all", experimento="E5")
        for avaliacao in ["in_sample", "holdout_80_20"]:
            treeClassification.run(target="ternario", recorte=recorte, avaliacao=avaliacao, experimento="E5")


def gerar_artigos_arvores_profundidade():
    from src.article import treeClassification_profundidade, apendice_treeClassification_profundidade

    print("=" * 80)
    print("PRODUTOS | ÁRVORES DE DECISÃO | PROFUNDIDADES 10, 14 E 19")
    print("=" * 80)

    for profundidade in [10, 14, 19]:
        print("-" * 80)
        print(f"PROFUNDIDADE {profundidade}")
        print("-" * 80)

        apendice_treeClassification_profundidade.run(
            target="binario",
            recorte="geral",
            avaliacao="all",
            experimento="E5",
            profundidade=profundidade,
        )
        for avaliacao in ["in_sample", "holdout_80_20"]:
            treeClassification_profundidade.run(
                target="binario",
                recorte="geral",
                avaliacao=avaliacao,
                experimento="E5",
                profundidade=profundidade,
            )

        for recorte in ["geral", "medicina"]:
            apendice_treeClassification_profundidade.run(
                target="ternario",
                recorte=recorte,
                avaliacao="all",
                experimento="E5",
                profundidade=profundidade,
            )
            for avaliacao in ["in_sample", "holdout_80_20"]:
                treeClassification_profundidade.run(
                    target="ternario",
                    recorte=recorte,
                    avaliacao=avaliacao,
                    experimento="E5",
                    profundidade=profundidade,
                )


def gerar_artigos_modelagem():
    gerar_artigos_logit()
    gerar_artigos_arvore_padrao()
    gerar_artigos_arvores_profundidade()


def executar_reproducao(etapa: str):
    refit = tem_flag("--refit") or tem_flag("--refit-models") or tem_flag("--retrain")

    if etapa in ["descriptive", "descritivo", "descritiva"]:
        executar_pipeline("all")
        executar_analysis("all")
        executar_artigos_descritivos()
        return

    if etapa == "all":
        executar_pipeline("all")
        executar_analysis("all")
        executar_artigos_descritivos()
        executar_abts_modelagem()
        if refit:
            treinar_todos_modelos(force=True)
        gerar_artigos_modelagem()
        # Ao final, organiza o pacote final em article/.
        # O export inclui os efeitos multinomiais usados nas seções 4.4 e 4.5.
        from src.article import pacote_artigo
        pacote_artigo.run(
            output_dir=valor_flag("--out", "article"),
            avaliacao=valor_flag("--avaliacao", "in_sample"),
            clean=True,
            dry_run=False,
        )
        return

    print(f"Etapa de reprodução não reconhecida: {etapa}")
    ajuda()


def executar_export(etapa: str):
    if etapa not in ["article", "artigo"]:
        print(f"Exportação não reconhecida: {etapa}")
        ajuda()
        return

    from src.article import pacote_artigo

    pacote_artigo.run(
        output_dir=valor_flag("--out", "article"),
        avaliacao=valor_flag("--avaliacao", "in_sample"),
        clean=tem_flag("--clean"),
        dry_run=tem_flag("--dry-run"),
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["help", "-h", "--help"]:
        ajuda()
        return

    comando = sys.argv[1]

    if comando == "init":
        from src.config import ensure_project_structure
        ensure_project_structure()
    elif comando == "check":
        from src.config import check_environment
        check_environment()
    elif comando == "pipeline":
        if len(sys.argv) < 3:
            ajuda()
            return
        executar_pipeline(sys.argv[2])
    elif comando == "transform":
        if len(sys.argv) < 3:
            ajuda()
            return
        executar_transform(sys.argv[2])
    elif comando == "analysis":
        if len(sys.argv) < 3:
            ajuda()
            return
        executar_analysis(sys.argv[2])
    elif comando == "abt":
        if len(sys.argv) < 3:
            ajuda()
            return
        executar_abt(sys.argv[2])
    elif comando == "modeling":
        if len(sys.argv) < 3:
            ajuda()
            return
        acao = sys.argv[2]
        alvo = sys.argv[3] if len(sys.argv) >= 4 and not sys.argv[3].startswith("--") else "general"
        executar_modelagem(acao, alvo)
    elif comando == "article":
        if len(sys.argv) < 3:
            ajuda()
            return
        executar_artigo(sys.argv[2])
    elif comando == "reproduce":
        if len(sys.argv) < 3:
            ajuda()
            return
        executar_reproducao(sys.argv[2])
    elif comando == "export":
        if len(sys.argv) < 3:
            ajuda()
            return
        executar_export(sys.argv[2])
    elif comando == "diagnostics":
        executar_diagnosticos()
    else:
        print(f"Comando não reconhecido: {comando}")
        ajuda()


if __name__ == "__main__":
    main()
