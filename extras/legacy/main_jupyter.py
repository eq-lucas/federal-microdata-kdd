# %%
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




comando = 'analise'
etapa = '11'

# --- CAMADA: STAGING ---
if comando == "staging":
    staging_fies()
    staging_inep()

# --- CAMADA: TRANSFORM ---
elif comando == "transform":

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

elif comando == "load":

    load_inscritos()
    load_ofertas()
    auditoria_inscritos_carregados()
    auditoria_ofertas_carregadas()
    exportar_para_sqlite()

elif comando == 'analise':

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

        # %%
