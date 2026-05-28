from pathlib import Path

# pasta raiz 
pasta_raiz_projeto = Path(__file__).resolve().parent.parent

# pasta data 01 raw:
pasta_data_01_raw_microdata_fies = pasta_raiz_projeto / 'data' / '01_raw' / 'microdata fies'
pasta_data_01_raw_microdata_inep = pasta_raiz_projeto / 'data' / '01_raw' / 'microdata inep' # REMOVIDO str()

# pasta data 02 staging:
pasta_data_02_staging_microdata_fies = pasta_raiz_projeto / 'data' / '02_staging' / 'microdata fies'
pasta_data_02_staging_microdata_fies_errors = pasta_data_02_staging_microdata_fies / 'errors'
pasta_data_02_staging_microdata_inep = pasta_raiz_projeto / 'data' / '02_staging' / 'microdata inep' # REMOVIDO str()

# pasta data 03 transform:
pasta_data_03_transform_inep = pasta_raiz_projeto / 'data' / '03_transform' / 'inep'
pasta_data_03_transform_fies = pasta_raiz_projeto / 'data' / '03_transform' / 'fies'
pasta_data_03_temporarios = pasta_raiz_projeto / 'data' / '03_transform' / 'temporarios'

# pasta data 04 load:
pasta_data_04_load = pasta_raiz_projeto / 'data' / '04_load'

pasta_data_04_load_database = pasta_data_04_load / 'database'



pasta_data_04_load_inscritos = pasta_data_04_load_database / 'inscritos_final_limpo.parquet'

pasta_data_04_load_ofertas = pasta_data_04_load_database / 'ofertas_final_limpo.parquet'



#pasta 05 processed:

pasta_data_05_processed = pasta_raiz_projeto / 'data' / '05_processed'

pasta_data_05_processed_candidatos_unicos_por_prioridade_inicial = pasta_data_05_processed / 'candidatos_unicos_por_prioridade_inicial.parquet'


# pasta 05 processed 07:


pasta_data_05_processed_analise_007_GAP_RENDA_AREA_CINE =  pasta_data_05_processed / 'analise_007'

pasta_data_05_processed_analise_007_X_treino = pasta_data_05_processed_analise_007_GAP_RENDA_AREA_CINE / 'x_treino.parquet'

pasta_data_05_processed_analise_007_X_teste = pasta_data_05_processed_analise_007_GAP_RENDA_AREA_CINE / 'x_teste.parquet'

pasta_data_05_processed_analise_007_y_treino = pasta_data_05_processed_analise_007_GAP_RENDA_AREA_CINE / 'y_treino.parquet'

pasta_data_05_processed_analise_007_y_teste = pasta_data_05_processed_analise_007_GAP_RENDA_AREA_CINE / 'y_teste.parquet'


#pasta 05 processed 08:


pasta_data_05_processed_analise_008_GAP_RENDA_AREA_CINE_CANDIDATOS_PRIORIDADE_INICIAL =  pasta_data_05_processed / 'analise_008'

pasta_data_05_processed_analise_008_X_treino = pasta_data_05_processed_analise_008_GAP_RENDA_AREA_CINE_CANDIDATOS_PRIORIDADE_INICIAL / 'x_treino.parquet'

pasta_data_05_processed_analise_008_X_teste = pasta_data_05_processed_analise_008_GAP_RENDA_AREA_CINE_CANDIDATOS_PRIORIDADE_INICIAL / 'x_teste.parquet'

pasta_data_05_processed_analise_008_y_treino = pasta_data_05_processed_analise_008_GAP_RENDA_AREA_CINE_CANDIDATOS_PRIORIDADE_INICIAL / 'y_treino.parquet'

pasta_data_05_processed_analise_008_y_teste = pasta_data_05_processed_analise_008_GAP_RENDA_AREA_CINE_CANDIDATOS_PRIORIDADE_INICIAL / 'y_teste.parquet'



#pasta 05 processed 09:


pasta_data_05_processed_analise_009 =  pasta_data_05_processed / 'analise_009'

pasta_data_05_processed_analise_009_X_treino = pasta_data_05_processed_analise_009 / 'x_treino.parquet'

pasta_data_05_processed_analise_009_X_teste = pasta_data_05_processed_analise_009 / 'x_teste.parquet'

pasta_data_05_processed_analise_009_y_treino = pasta_data_05_processed_analise_009 / 'y_treino.parquet'

pasta_data_05_processed_analise_009_y_teste = pasta_data_05_processed_analise_009 / 'y_teste.parquet'


#pasta 05 processed 09_1:


pasta_data_05_processed_analise_009_1 =  pasta_data_05_processed / 'analise_009_1'

pasta_data_05_processed_analise_009_1_X_treino = pasta_data_05_processed_analise_009_1 / 'x_treino.parquet'

pasta_data_05_processed_analise_009_1_X_teste = pasta_data_05_processed_analise_009_1 / 'x_teste.parquet'

pasta_data_05_processed_analise_009_1_y_treino = pasta_data_05_processed_analise_009_1 / 'y_treino.parquet'

pasta_data_05_processed_analise_009_1_y_teste = pasta_data_05_processed_analise_009_1 / 'y_teste.parquet'

#pasta 05 processed 10:


pasta_data_05_processed_analise_010 =  pasta_data_05_processed / 'analise_010'

pasta_data_05_processed_analise_010_X_treino = pasta_data_05_processed_analise_010 / 'x_treino.parquet'

pasta_data_05_processed_analise_010_X_teste = pasta_data_05_processed_analise_010 / 'x_teste.parquet'

pasta_data_05_processed_analise_010_y_treino = pasta_data_05_processed_analise_010 / 'y_treino.parquet'

pasta_data_05_processed_analise_010_y_teste = pasta_data_05_processed_analise_010 / 'y_teste.parquet'

#pasta 05 processed 12_1 ( medicina todos os anos )

pasta_data_05_processed_analise_012_1 = pasta_data_05_processed / 'analise_012_1'

pasta_data_05_processed_analise_012_1_X_treino = pasta_data_05_processed_analise_012_1 / "x_treino.parquet"

pasta_data_05_processed_analise_012_1_y_treino = pasta_data_05_processed_analise_012_1 / "y_treino.parquet"

#pasta 05 processed 13:


pasta_data_05_processed_analise_013 =  pasta_data_05_processed / 'analise_013'

pasta_data_05_processed_analise_013_X_treino = pasta_data_05_processed_analise_013 / 'x_treino.parquet'

pasta_data_05_processed_analise_013_X_teste = pasta_data_05_processed_analise_013 / 'x_teste.parquet'

pasta_data_05_processed_analise_013_y_treino = pasta_data_05_processed_analise_013 / 'y_treino.parquet'

pasta_data_05_processed_analise_013_y_teste = pasta_data_05_processed_analise_013 / 'y_teste.parquet'


#pasta 05 processed 14:


pasta_data_05_processed_analise_014 =  pasta_data_05_processed / 'analise_014'

pasta_data_05_processed_analise_014_X_treino = pasta_data_05_processed_analise_014 / 'x_treino.parquet'

pasta_data_05_processed_analise_014_X_teste = pasta_data_05_processed_analise_014 / 'x_teste.parquet'

pasta_data_05_processed_analise_014_y_treino = pasta_data_05_processed_analise_014 / 'y_treino.parquet'

pasta_data_05_processed_analise_014_y_teste = pasta_data_05_processed_analise_014 / 'y_teste.parquet'

#pasta 05 processed 15:


pasta_data_05_processed_analise_015 =  pasta_data_05_processed / 'analise_015'

pasta_data_05_processed_analise_015_X_treino = pasta_data_05_processed_analise_015 / 'x_treino.parquet'

pasta_data_05_processed_analise_015_X_teste = pasta_data_05_processed_analise_015 / 'x_teste.parquet'

pasta_data_05_processed_analise_015_y_treino = pasta_data_05_processed_analise_015 / 'y_treino.parquet'

pasta_data_05_processed_analise_015_y_teste = pasta_data_05_processed_analise_015 / 'y_teste.parquet'

#pasta 05 processed 16:


pasta_data_05_processed_analise_016 =  pasta_data_05_processed / 'analise_016'

pasta_data_05_processed_analise_016_X_treino = pasta_data_05_processed_analise_016 / 'x_treino.parquet'

pasta_data_05_processed_analise_016_X_teste = pasta_data_05_processed_analise_016 / 'x_teste.parquet'

pasta_data_05_processed_analise_016_y_treino = pasta_data_05_processed_analise_016 / 'y_treino.parquet'

pasta_data_05_processed_analise_016_y_teste = pasta_data_05_processed_analise_016 / 'y_teste.parquet'

#pasta 05 processed 17:


pasta_data_05_processed_analise_017 =  pasta_data_05_processed / 'analise_017'

pasta_data_05_processed_analise_017_X_treino = pasta_data_05_processed_analise_017 / 'x_treino.parquet'

pasta_data_05_processed_analise_017_X_teste = pasta_data_05_processed_analise_017 / 'x_teste.parquet'

pasta_data_05_processed_analise_017_y_treino = pasta_data_05_processed_analise_017 / 'y_treino.parquet'

pasta_data_05_processed_analise_017_y_teste = pasta_data_05_processed_analise_017 / 'y_teste.parquet'

#pasta 05 processed 18:


pasta_data_05_processed_analise_018 =  pasta_data_05_processed / 'analise_018'

pasta_data_05_processed_analise_018_X_treino = pasta_data_05_processed_analise_018 / 'x_treino.parquet'

pasta_data_05_processed_analise_018_X_teste = pasta_data_05_processed_analise_018 / 'x_teste.parquet'

pasta_data_05_processed_analise_018_y_treino = pasta_data_05_processed_analise_018 / 'y_treino.parquet'

pasta_data_05_processed_analise_018_y_teste = pasta_data_05_processed_analise_018 / 'y_teste.parquet'

#pasta 05 processed 19:


pasta_data_05_processed_analise_019 =  pasta_data_05_processed / 'analise_019'

pasta_data_05_processed_analise_019_X_treino = pasta_data_05_processed_analise_019 / 'x_treino.parquet'

pasta_data_05_processed_analise_019_X_teste = pasta_data_05_processed_analise_019 / 'x_teste.parquet'

pasta_data_05_processed_analise_019_y_treino = pasta_data_05_processed_analise_019 / 'y_treino.parquet'

pasta_data_05_processed_analise_019_y_teste = pasta_data_05_processed_analise_019 / 'y_teste.parquet'


# pasta modelo:

pasta_modelo = pasta_raiz_projeto / 'reports' / 'models'

pasta_modelo_analise_007 =  pasta_modelo / 'analise_007_modelo_fies_logistica.pkl'

pasta_modelo_analise_008 = pasta_modelo / 'analise_008_modelo_fies_logistica.pkl'

pasta_modelo_analise_009 = pasta_modelo / 'analise_009_modelo_fies_logistica.pkl'

pasta_modelo_analise_010 = pasta_modelo / 'analise_010_modelo_fies_logistica.pkl'

pasta_modelo_analise_009_1 = pasta_modelo / 'analise_009_1_modelo_fies_logistica.pkl'

pasta_modelo_analise_013 = pasta_modelo / 'analise_013_modelo_fies_logistica.pkl'

pasta_modelo_analise_014 = pasta_modelo / 'analise_014_modelo_fies_logistica.pkl'

pasta_modelo_analise_015 = pasta_modelo / 'analise_015_modelo_fies_logistica.pkl'

pasta_modelo_analise_016 = pasta_modelo / 'analise_016_modelo_fies_logistica.pkl'

pasta_modelo_analise_017 = pasta_modelo / 'analise_017_modelo_fies_logistica.pkl'

pasta_modelo_analise_018 = pasta_modelo / 'analise_018_modelo_fies_logistica.pkl'

pasta_modelo_analise_019 = pasta_modelo / 'analise_019_modelo_fies_logistica.pkl'







