import sys
import pandas as pd
from src.staging import staging_fies, staging_inep
from src.transform_layer_1 import transform_inscritos, transform_ofertas, transform_inep, verificar_colunas_inep
from src.transform_layer_2 import tratar_nans_cine_inscritos, tratar_nans_cine_ofertas
from src.trasnform_layer_3 import processar_modalidades_e_peneira
from src.load import load_inscritos,load_ofertas,auditoria_inscritos_carregados,auditoria_ofertas_carregadas,exportar_para_sqlite
from src.analise_001 import gerar_dataset_candidatos_unicos_por_Prioridade_inicial
from src.analise_002 import eda_gerar_dataset_com_varias_analises
from src.analise_003 import eda_gerar_funil_de_selecao_6_etapas
from src.analise_003_1 import eda_gerar_funil_de_selecao_6_etapas_v2
from src.analise_004 import analise_fuga_cerebros_de_regioes
from src.analise_005 import verificar_qtde_inscritos_por_ano_e_semestre_nacional_e_regional
from src.analise_006 import gerar_dataset_e_grafico_heatmap_quem_sao_os_inscritos_contratados
from src.analise_007 import orquestrador_inicial_inscritos, orquestrador_ja_rodado_inscritos
from src.analise_008 import orquestrador_inicial_candidatos, orquestrador_ja_rodado_candidatos
from src.analise_009 import orquestrador_inicial_inscritos_nove, orquestrador_ja_rodado_inscritos_nove
from src.analise_009_1 import orquestrador_inicial_inscritos_nove_um, orquestrador_ja_rodado_inscritos_nove_um
from src.analise_010 import orquestrador_inicial_inscritos_dez, orquestrador_ja_rodado_inscritos_dez
from src.analise_011 import orquestrador_analise_011
from src.analise_013 import orquestrador_inicial_inscritos_13,orquestrador_ja_rodado_inscritos_13
from src.analise_014 import orquestrador_inicial_inscritos_14,orquestrador_ja_rodado_inscritos_14
from src.analise_015 import orquestrador_inicial_inscritos_15,orquestrador_ja_rodado_inscritos_15
from src.analise_016 import orquestrador_inicial_inscritos_16,orquestrador_ja_rodado_inscritos_16
from src.analise_017_0_com_insights import orquestrador_inicial_inscritos_17,orquestrador_ja_rodado_inscritos_17
from src.analise_018 import orquestrador_inicial_inscritos_18,orquestrador_ja_rodado_inscritos_18
from src.analise_019 import orquestrador_inicial_inscritos_19,orquestrador_ja_rodado_inscritos_19


def imprimir_ajuda():
    print("\n" + "="*85)
    print(" PIPELINE FIES KDD - MENU DE AJUDA")
    print("="*85)
    print("Uso: python3 main.py [comando] [etapa]")
    print("\nCOMANDOS DE STAGING:")
    print("  staging              : Executa toda a limpeza inicial (FIES e INEP)")
    print("\nCOMANDOS DE TRANSFORM (POR CAMADAS):")
    print("  transform 1          : Etapa 1 - Gera Dataset Mestre INEP")
    print("  transform 2          : Etapa 2 - Unifica Inscritos (FIES)")
    print("  transform 3          : Etapa 3 - Unifica Ofertas (FIES)")
    print("  transform 4          : Etapa 4 - Auditoria de Colunas INEP")
    print("  transform 5          : Etapa 5 - Trata NaNs CINE (Inscritos)")
    print("  transform 6          : Etapa 6 - Trata NaNs CINE (Ofertas)")
    print("  transform 7          : Etapa 7 - Processa Modalidades e Peneira de Renda")
    print("  transform all        : Executa todas as transformações em sequência")
    print("\nCOMANDOS DE LOAD:")
    print("  load                 : Etapa 8 - Padronização Final (ABT) para Banco de Dados")
    print("="*85 + "\n")
    print("\nCOMANDOS DE ANALISE (KDD):")
    print("  analise 1            : Candidatos Únicos por Prioridade Inicial")
    print("  analise 2            : EDA - Dataset com Múltiplas Visões de Análise")
    print("  analise 3            : Dataset Funil de Seleção de 6 Etapas")
    print("  analise 3.1          : Funil de Seleção de 6 Etapas | o de inscritos e o de candidatos de prioridade inicial")
    print("  analise 4            : Análise de Fuga de Cérebros (Fluxo Regional)")
    print("  analise 5            : Volumetria Nacional e Regional por Ano/Semestre")
    print("  analise 6            : EDA para saber quem esta sendo efetivametne contratado em relação GAP e renda")
    print("  analise 6.1          : (APENAS POR JUPYTER NOTEBOOK) Insights respectivos da matriz como inscritos em cada grupo de renda x GAP")
    print("  analise 7            : ML Orquestrador (Renda vs GAP vs CINE) para inscritos")
    print("  analise 7.1          : ML Orquestrador (Renda vs GAP vs CINE) se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 8            : ML Orquestrador (Renda vs GAP vs CINE) para candidatos unicos de prioridade inicial")
    print("  analise 8.1          : ML Orquestrador (Renda vs GAP vs CINE) se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 9            : ML Orquestrador Multivariado (ElasticNet) para inscritos")
    print("  analise 9.1          : ML Orquestrador Multivariado (ElasticNet) se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 9_1          : Estudo de Ablação (ElasticNet SEM Opção de Curso) para inscritos")
    print("  analise 9_1.1        : Estudo de Ablação se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 10           : ML Orquestrador Multivariado (ElasticNet) para candidatos unicos de prioridade inicial")
    print("  analise 10.1         : ML Orquestrador Multivariado (ElasticNet) se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 11           : Regressão Linear e Correlação (Renda vs Percentual de Financiamento)")
    print("  analise 12           : (APENAS POR JUPYTER NOTEBOOK) ML Orquestrador RandomForest para inscritos")
    print("  analise 12.1         : (APENAS POR JUPYTER NOTEBOOK) ML Orquestrador RandomForest para inscritos")
    print("  analise 13           : ML Orquestrador Multivariado como a analise009 mas sem penalty para inscritos")
    print("  analise 13.1         : ML Orquestrador Multivariado, se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 14           : ML Orquestrador Multivariado usufruindo tambem de modalidade fies e outras, e efeitos probabilisticos para inscritos")
    print("  analise 14.1         : ML Orquestrador Multivariado, se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 15           : ML Orquestrador Multivariado como analise14 porem somente aos de medicina e para todo o ABT  (ANO e semestre)")
    print("  analise 15.1         : ML Orquestrador Multivariado, se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 16           : ML Orquestrador Multivariado como analise15 mas sem opcoes como metrica para ML")
    print("  analise 16.1         : ML Orquestrador Multivariado, se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 17           : ML Orquestrador Multivariado como analise14 porem usando df treino e sem distincao de ano/semestre para teste")
    print("  analise 17.1         : ML Orquestrador Multivariado, se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 17.1.1       : (APENAS POR JUPYTER NOTEBOOK) ML Orquestrador Multivariado como analise17 porem para medicina")
    print("  analise 18           : ML Orquestrador Multivariado como analise17 porem apenas para Medicina e Y ternario")
    print("  analise 18.1         : ML Orquestrador Multivariado, se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 19           : ML Orquestrador Multivariado como analise17 mas com y_target ternario (incluindo Lista de espera)")
    print("  analise 19.1         : ML Orquestrador Multivariado, se ja rodou uma vez para nao re-criar e gerar modelo dnv")
    print("  analise 20           : teste desconsideravel para analise de modelo")
    print("  analise 21           : (APENAS POR JUPYTER NOTEBOOK)geracao temporaria do dataset de taxas e geracao dos graficos de taxas")
    print("  analise 22           : (APENAS POR JUPYTER NOTEBOOK)geracao temporaria do dataset de qtde inscritos / situacao inscricao fies")

    


    print("="*85 + "\n")

def main():
    
    # Se não passar argumentos ou pedir help
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        imprimir_ajuda()
        return

    comando = sys.argv[1]

    # --- CAMADA: STAGING ---
    if comando == "staging":
        staging_fies()
        staging_inep()

    # --- CAMADA: TRANSFORM ---
    elif comando == "transform":

        if len(sys.argv) < 3:
            print("\n Erro: O comando 'transform' precisa de uma etapa específica.")
            print("Exemplo: python3 main.py transform 1")
            return

        etapa = sys.argv[2]

        if etapa == "1":
            transform_inep()
        elif etapa == "2":
            transform_inscritos()
        elif etapa == "3":
            transform_ofertas()
        elif etapa == "4":
            verificar_colunas_inep()
        elif etapa == "5":
            tratar_nans_cine_inscritos()
        elif etapa == "6":
            tratar_nans_cine_ofertas()
        elif etapa == "7":
            processar_modalidades_e_peneira()
            
        elif etapa == "all":
            transform_inep()
            transform_inscritos()
            transform_ofertas()
            verificar_colunas_inep()
            tratar_nans_cine_inscritos()
            tratar_nans_cine_ofertas()
            processar_modalidades_e_peneira()
        else:
            print(f"\n Etapa '{etapa}' não reconhecida.")
            imprimir_ajuda()

    # --- CAMADA: LOAD ---
    elif comando == "load":

        load_inscritos()
        load_ofertas()
        auditoria_inscritos_carregados()
        auditoria_ofertas_carregadas()
        exportar_para_sqlite()

    # ANALISES
    elif comando == 'analise':

        etapa = sys.argv[2]

        if len(sys.argv) < 3:
            print("\n Erro: O comando 'analise' precisa de uma etapa específica (1-7).")
            print("Exemplo: python3 main.py analise 7")
            return

        if etapa == "1":
            gerar_dataset_candidatos_unicos_por_Prioridade_inicial()

        elif etapa == "2":
            eda_gerar_dataset_com_varias_analises()

        elif etapa == "3":
            eda_gerar_funil_de_selecao_6_etapas()

        elif etapa == "3.1":
            eda_gerar_funil_de_selecao_6_etapas_v2()

        elif etapa == "4":
            analise_fuga_cerebros_de_regioes()

        elif etapa == "5":
            verificar_qtde_inscritos_por_ano_e_semestre_nacional_e_regional()

        elif etapa == "6":
            gerar_dataset_e_grafico_heatmap_quem_sao_os_inscritos_contratados()

        elif etapa == "7":
            orquestrador_inicial_inscritos()

        elif etapa == "7.1":
            orquestrador_ja_rodado_inscritos()

        elif etapa == "8":
            orquestrador_inicial_candidatos()
        
        elif etapa == "8.1":
            orquestrador_ja_rodado_candidatos()

        elif etapa == "9":
            orquestrador_inicial_inscritos_nove()
        
        elif etapa == "9.1":
            orquestrador_ja_rodado_inscritos_nove()

        elif etapa == "9_1":
            orquestrador_inicial_inscritos_nove_um()
        
        elif etapa == "9_1.1":
            orquestrador_ja_rodado_inscritos_nove_um()

        elif etapa == "10":
            orquestrador_inicial_inscritos_dez()
        
        elif etapa == "10.1":
            orquestrador_ja_rodado_inscritos_dez()

        elif etapa == "11":
            orquestrador_analise_011()

        elif etapa == "13":
            orquestrador_inicial_inscritos_13()
        
        elif etapa == "13.1":
            orquestrador_ja_rodado_inscritos_13()
        

        elif etapa == "14":
            orquestrador_inicial_inscritos_14()
        
        elif etapa == "14.1":
            orquestrador_ja_rodado_inscritos_14()

        elif etapa == "15":
            orquestrador_inicial_inscritos_15()
        
        elif etapa == "15.1":
            orquestrador_ja_rodado_inscritos_15()

        elif etapa == "16":
            orquestrador_inicial_inscritos_16()
        
        elif etapa == "16.1":
            orquestrador_ja_rodado_inscritos_16()

        elif etapa == "17":
            orquestrador_inicial_inscritos_17()
        
        elif etapa == "17.1":
            orquestrador_ja_rodado_inscritos_17()

        elif etapa == "18":
            orquestrador_inicial_inscritos_18()
    
        elif etapa == "18.1":
            orquestrador_ja_rodado_inscritos_18()

        elif etapa == "19":
            orquestrador_inicial_inscritos_19()
    
        elif etapa == "19.1":
            orquestrador_ja_rodado_inscritos_19()


        else:
            print(f"\n Etapa '{etapa}' não reconhecida.")
            imprimir_ajuda()


    else:
        print(f"\n Comando '{comando}' inválido.")
        imprimir_ajuda()

if __name__ == "__main__":
    main()