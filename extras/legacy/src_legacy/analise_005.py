def verificar_qtde_inscritos_por_ano_e_semestre_nacional_e_regional():

    # visualizacao qtde de inscritos em cada ano/ periodo e em regiao
    import pandas as pd
    from src.constantes import pasta_data_04_load_inscritos

    # Força o Pandas a mostrar todas as linhas para você conseguir rolar e ver todos os estados
    pd.set_option('display.max_rows', 300)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n" + "="*90)
    print("🗺️ MAPA DE DISTRIBUIÇÃO GEOGRÁFICA DO FIES (DESTINO / IES)")
    print("="*90)

    # --- 1. CARREGAR OS DADOS ---
    print("[*] Carregando a base Load de Inscritos...\n")
    df = pd.read_parquet(str(pasta_data_04_load_inscritos))

    # Verifica se a coluna corrigida existe (senão, usa a original)
    coluna_uf = 'uf_ies_corrigida' if 'uf_ies_corrigida' in df.columns else 'uf_ies'

    # --- 2. VISÃO MACRO: PIVOT TABLE POR REGIÃO ALVO ---
    print("-" * 90)
    print("📊 1. VISÃO MACRO: INSCRITOS POR REGIÃO DA IES (Ano e Semestre)")
    print("-" * 90)

    pivot_regiao = df.pivot_table(
        index=['ano', 'semestre'], 
        columns='regiao_ies_alvo', 
        aggfunc='size', 
        fill_value=0
    )

    # Adiciona uma coluna de Total para você ver que a matemática bate com os 2.2 milhões
    pivot_regiao['TOTAL_BRASIL'] = pivot_regiao.sum(axis=1)
    print(pivot_regiao.to_string())


    # --- 3. VISÃO MICRO: VOLUMETRIA POR UF ALVO ---
    print("\n\n" + "-" * 90)
    print(f"📊 2. VISÃO MICRO: INSCRITOS POR UF DE DESTINO ({coluna_uf})")
    print("-" * 90)

    resumo_uf = (df.groupby(['ano', 'semestre', 'regiao_ies_alvo', coluna_uf])
                .size()
                .reset_index(name='Total_Inscritos'))

    # Ordena por Ano, Semestre, e depois do estado com MAIS inscritos para o com MENOS
    resumo_uf = resumo_uf.sort_values(by=['ano', 'semestre', 'Total_Inscritos'], ascending=[True, True, False])

    print(resumo_uf.to_string(index=False))

    print("\n" + "="*90)
    print("✅ Agrupamento concluído! Verifique se todos os estados estão com o volume esperado.")
    print("="*90 + "\n")
    # %%