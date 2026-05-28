import pandas as pd
import numpy as np
from pathlib import Path
from src.constantes import pasta_data_03_temporarios,pasta_raiz_projeto,pasta_data_02_staging_microdata_fies,pasta_data_01_raw_microdata_fies,pasta_data_02_staging_microdata_inep,pasta_data_03_transform_inep,pasta_data_03_transform_fies

def tratar_nans_cine_inscritos():
    print("\n" + "="*60)
    print("INICIANDO LAYER 2: TRATAMENTO DE NaNs CINE - INSCRITOS")
    print("="*60)

    arquivo_entrada = pasta_data_03_transform_fies / 'fies_inscritos_unificado.parquet'
    
    # Nomes das colunas
    col_chave = 'nome_curso_inscricao'
    col_val_nome = 'NO_CINE_AREA_GERAL'
    col_val_cod = 'CO_CINE_AREA_GERAL'

    # --- 2. MAPA DE CORREÇÃO MANUAL ---
    # Coloque aqui todos os cursos que você mapeou manualmente
    mapa_correcao_manual = {
    'DIREITO': ('Negócios, administração e direito', '04'),
    'ENFERMAGEM': ('Saúde e bem-estar', '09'),
    'ADMINISTRAÇÃO': ('Negócios, administração e direito', '04'),
    'NUTRIÇÃO': ('Saúde e bem-estar', '09'),
    'PSICOLOGIA': ('Ciências sociais, comunicação e informação', '03'),
    'FARMÁCIA': ('Saúde e bem-estar', '09'),
    'FISIOTERAPIA': ('Saúde e bem-estar', '09'),
    'EDUCAÇÃO FÍSICA': ('Educação', '01'),
    'PEDAGOGIA': ('Educação', '01'),
    'CIÊNCIAS CONTÁBEIS': ('Negócios, administração e direito', '04'),
    'GESTÃO DE RECURSOS HUMANOS': ('Negócios, administração e direito', '04'),
    'ARQUITETURA E URBANISMO': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA CIVIL': ('Engenharia, produção e construção', '07'),
    'ODONTOLOGIA': ('Saúde e bem-estar', '09'),
    'BIOMEDICINA': ('Saúde e bem-estar', '09'),
    'ENGENHARIA ELÉTRICA': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE PRODUÇÃO': ('Engenharia, produção e construção', '07'),
    'SERVIÇO SOCIAL': ('Saúde e bem-estar', '09'),
    'LOGÍSTICA': ('Serviços', '10'),
    'MEDICINA VETERINÁRIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'ENGENHARIA MECÂNICA': ('Engenharia, produção e construção', '07'),
    'ANÁLISE E DESENVOLVIMENTO DE SISTEMAS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'RADIOLOGIA': ('Saúde e bem-estar', '09'),
    'SISTEMAS DE INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'CIÊNCIAS BIOLÓGICAS': ('Ciências naturais, matemática e estatística', '05'),
    'MARKETING': ('Negócios, administração e direito', '04'),
    'GESTÃO FINANCEIRA': ('Negócios, administração e direito', '04'),
    'GESTÃO COMERCIAL': ('Negócios, administração e direito', '04'),
    'GASTRONOMIA': ('Serviços', '10'),
    'ESTÉTICA E COSMÉTICA': ('Serviços', '10'),
    'GESTÃO HOSPITALAR': ('Negócios, administração e direito', '04'),
    'REDES DE COMPUTADORES': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'SEGURANÇA NO TRABALHO': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE COMPUTAÇÃO': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA AMBIENTAL E SANITÁRIA': ('Engenharia, produção e construção', '07'),
    'GESTÃO DE SEGURANÇA PRIVADA': ('Serviços', '10'),
    'ENGENHARIA DE CONTROLE E AUTOMAÇÃO': ('Engenharia, produção e construção', '07'),
    'HISTÓRIA': ('Educação', '01'),
    'CONSTRUÇÃO DE EDIFÍCIOS': ('Engenharia, produção e construção', '07'),
    'GESTÃO AMBIENTAL': ('Negócios, administração e direito', '04'),
    'JORNALISMO': ('Ciências sociais, comunicação e informação', '03'),
    'DESIGN GRÁFICO': ('Artes e humanidades', '02'),
    'GESTÃO PÚBLICA': ('Negócios, administração e direito', '04'),
    'DESIGN DE INTERIORES': ('Artes e humanidades', '02'),
    'PROCESSOS GERENCIAIS': ('Negócios, administração e direito', '04'),
    'GESTÃO DA TECNOLOGIA DA INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'LETRAS - LÍNGUA PORTUGUESA': ('Educação', '01'),
    'PUBLICIDADE E PROPAGANDA': ('Ciências sociais, comunicação e informação', '03'),
    'ENGENHARIA QUÍMICA': ('Engenharia, produção e construção', '07'),
    'CIÊNCIA DA COMPUTAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'LETRAS - PORTUGUÊS': ('Educação', '01'),
    'SISTEMAS PARA INTERNET': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'MATEMÁTICA': ('Educação', '01'),
    'TURISMO': ('Serviços', '10'),
    'GEOGRAFIA': ('Educação', '01'),
    'LETRAS - INGLÊS': ('Educação', '01'),
    'COMÉRCIO EXTERIOR': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO SOCIAL - PUBLICIDADE E PROPAGANDA': ('Negócios, administração e direito', '04'),
    'ENGENHARIA AGRONÔMICA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'LETRAS - PORTUGUÊS E INGLÊS': ('Educação', '01'),
    'RELAÇÕES INTERNACIONAIS': ('Ciências sociais, comunicação e informação', '03'),
    'SECRETARIADO EXECUTIVO': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO SOCIAL - JORNALISMO': ('Ciências sociais, comunicação e informação', '03'),
    'SISTEMA DE INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'GESTÃO DA QUALIDADE': ('Negócios, administração e direito', '04'),
    'ENGENHARIA AMBIENTAL': ('Engenharia, produção e construção', '07'),
    'JOGOS DIGITAIS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'TERAPIA OCUPACIONAL': ('Saúde e bem-estar', '09'),
    'DEFESA CIBERNÉTICA': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'SEGURANÇA DA INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'LETRAS - ESPANHOL': ('Educação', '01'),
    'AGRONOMIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'CIÊNCIAS ECONÔMICAS': ('Ciências sociais, comunicação e informação', '03'),
    'PRODUÇÃO MULTIMÍDIA': ('Artes e humanidades', '02'),
    'QUÍMICA': ('Educação', '01'),
    'CIÊNCIAS SOCIAIS': ('Educação', '01'),
    'ENGENHARIA AGRÍCOLA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'ENGENHARIA DE ENERGIA': ('Engenharia, produção e construção', '07'),
    'COMUNICAÇÃO SOCIAL': ('Ciências sociais, comunicação e informação', '03'),
    'NEGÓCIOS IMOBILIÁRIOS': ('Negócios, administração e direito', '04'),
    'LETRAS': ('Educação', '01'),
    'FOTOGRAFIA': ('Artes e humanidades', '02'),
    'PRODUÇÃO AUDIOVISUAL': ('Artes e humanidades', '02'),
    'RELAÇÕES PÚBLICAS': ('Ciências sociais, comunicação e informação', '03'),
    'PILOTAGEM PROFISSIONAL DE AERONAVES': ('Serviços', '10'),
    'ENGENHARIA MECATRÔNICA': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE TELECOMUNICAÇÕES': ('Engenharia, produção e construção', '07'),
    'GESTÃO DE TURISMO': ('Serviços', '10'),
    'ARTES VISUAIS': ('Educação', '01'),
    'BIOLOGIA': ('Ciências naturais, matemática e estatística', '05'),
    'PETRÓLEO E GÁS': ('Engenharia, produção e construção', '07'),
    'SECRETARIADO': ('Negócios, administração e direito', '04'),
    'ARTES CÊNICAS': ('Artes e humanidades', '02'),
    'FONOAUDIOLOGIA': ('Saúde e bem-estar', '09'),
    'ENGENHARIA DE PETRÓLEO': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE ALIMENTOS': ('Engenharia, produção e construção', '07'),
    'PROPAGANDA E MARKETING': ('Negócios, administração e direito', '04'),
    'CONSTRUÇÃO NAVAL': ('Engenharia, produção e construção', '07'),
    'FÍSICA': ('Educação', '01'),
    'HOTELARIA': ('Serviços', '10'),
    'SERVIÇOS PENAIS': ('Serviços', '10'),
    'AUTOMAÇÃO INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'LETRAS - PORTUGUÊS E ESPANHOL': ('Educação', '01'),
    'ENGENHARIA DA COMPUTAÇÃO': ('Engenharia, produção e construção', '07'),
    'CIÊNCIA POLÍTICA': ('Ciências sociais, comunicação e informação', '03'),
    'EVENTOS': ('Serviços', '10'),
    'TEOLOGIA': ('Artes e humanidades', '02'),
    'CIÊNCIA ECONÔMICA': ('Ciências sociais, comunicação e informação', '03'),
    'GEOPROCESSAMENTO': ('Engenharia, produção e construção', '07'),
    'CIÊNCIAS DA COMPUTAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'GESTÃO PORTUÁRIA': ('Serviços', '10'),
    'FORMAÇÃO PEDAGÓGICA PARA PORTADORES DE ENSINO SUPERIOR': ('Educação', '01'),
    'BANCO DE DADOS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'DESIGN DE MODA': ('Artes e humanidades', '02'),
    'GESTÃO DE EMPREENDIMENTOS ESPORTIVOS': ('Serviços', '10'),
    'PSICOPEDAGOGIA': ('Educação', '01'),
    'ENGENHARIA': ('Engenharia, produção e construção', '07'),
    'GESTÃO DA PRODUÇÃO INDUSTRIAL': ('Negócios, administração e direito', '04'),
    'DESIGN': ('Artes e humanidades', '02'),
    'FILOSOFIA': ('Educação', '01'),
    'MÚSICA': ('Artes e humanidades', '02'),
    'CINEMA E AUDIOVISUAL': ('Artes e humanidades', '02'),
    'COMUNICAÇÃO INSTITUCIONAL': ('Ciências sociais, comunicação e informação', '03'),
    'DESIGN DE PRODUTO': ('Artes e humanidades', '02'),
    'MARKETING DIGITAL': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO SOCIAL - CINEMA E AUDIOVISUAL': ('Artes e humanidades', '02'),
    'ENGENHARIA ELETRÔNICA': ('Engenharia, produção e construção', '07'),
    'ADMINISTRAÇÃO PÚBLICA': ('Negócios, administração e direito', '04'),
    'MANUTENÇÃO INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE PETRÓLEO E GÁS': ('Engenharia, produção e construção', '07'),
    'COMUNICAÇÃO SOCIAL COM HABILITAÇÃO EM PUBLICIDADE E PROPAGANDA': ('Ciências sociais, comunicação e informação', '03'),
    'GESTÃO DE SERVIÇOS JURÍDICOS, NOTARIAIS E DE REGISTRO': ('Negócios, administração e direito', '04'),
    'SISTEMAS ELÉTRICOS': ('Engenharia, produção e construção', '07'),
    'VISAGISMO E TERAPIAS CAPILARES': ('Serviços', '10'),
    'GERENCIAMENTO DE REDES DE COMPUTADORES': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'GESTÃO DE RECURSOS HÍDRICOS': ('Engenharia, produção e construção', '07'),
    'COMUNICAÇÃO EMPRESARIAL': ('Ciências sociais, comunicação e informação', '03'),
    'COMUNICAÇÃO SOCIAL - RADIO E TELEVISÃO': ('Ciências sociais, comunicação e informação', '03'),
    'ENERGIAS RENOVÁVEIS': ('Engenharia, produção e construção', '07'),
    'MODA': ('Artes e humanidades', '02'),
    'ENGENHARIA BIOMÉDICA': ('Engenharia, produção e construção', '07'),
    'SEGURANÇA PÚBLICA': ('Serviços', '10'),
    'CIÊNCIA DE DADOS E MACHINE LEARNING': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'GESTÃO DE PRODUÇÃO INDUSTRIAL': ('Negócios, administração e direito', '04'),
    'MULTIMÍDIA': ('Artes e humanidades', '02'),
    'BIOCOMBUSTÍVEIS': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE BIOPROCESSOS E BIOTECNOLOGIA': ('Engenharia, produção e construção', '07'),
    'ZOOTECNIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'GESTÃO DESPORTIVA E DE LAZER': ('Serviços', '10'),
    'TRANSPORTE TERRESTRE': ('Serviços', '10'),
    'PROCESSOS QUÍMICOS': ('Engenharia, produção e construção', '07'),
    'SECRETARIADO EXECUTIVO TRILINGUE': ('Negócios, administração e direito', '04'),
    'ELETROTÉCNICA INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'AGROPECUÁRIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'BIBLIOTECONOMIA': ('Ciências sociais, comunicação e informação', '03'),
    'AGRIMENSURA': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE SOFTWARE': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'ESTÉTICA': ('Serviços', '10'),
    'PRODUÇÃO GRÁFICA DIGITAL': ('Artes e humanidades', '02'),
    'TECNÓLOGO EM METALURGIA': ('Engenharia, produção e construção', '07'),
    'GESTÃO DE COOPERATIVAS': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO SOCIAL - RELAÇÕES PÚBLICAS': ('Ciências sociais, comunicação e informação', '03'),
    'ENGENHARIA DE MATERIAIS': ('Engenharia, produção e construção', '07'),
    'DANÇA': ('Artes e humanidades', '02'),
    'SEGURANÇA PRIVADA': ('Serviços', '10'),
    'MECATRÔNICA INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'POLÍTICAS PÚBLICAS': ('Ciências sociais, comunicação e informação', '03'),
    'PAPEL E CELULOSE': ('Engenharia, produção e construção', '07'),
    'SERVIÇOS JURÍDICOS, CARTORÁRIOS E NOTARIAIS': ('Negócios, administração e direito', '04'),
    'ENGENHARIA AEROESPACIAL': ('Engenharia, produção e construção', '07'),
    'AGRONEGÓCIO': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'PROCESSOS ESCOLARES': ('Educação', '01'),
    'LETRAS - LÍNGUA PORTUGUESA E LITERATURAS DE LÍNGUA PORTUGUESA': ('Educação', '01'),
    'PRODUÇÃO CULTURAL': ('Artes e humanidades', '02'),
    'GESTÃO DE SERVIÇOS JURÍDICOS E NOTARIAIS': ('Negócios, administração e direito', '04'),
    'GESTÃO DE NEGÓCIOS NO VAREJO': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO PARA WEB': ('Ciências sociais, comunicação e informação', '03'),
    'COMUNICAÇÃO SOCIAL\xa0 - RADIALISMO': ('Ciências sociais, comunicação e informação', '03'),
    'ACUPUNTURA': ('Saúde e bem-estar', '09'),
    'DESIGN DE ANIMAÇÃO': ('Artes e humanidades', '02'),
    'EDUCAÇÃO ARTÍSTICA': ('Educação', '01'),
    'GESTÃO DE NEGÓCIOS E INOVAÇÃO': ('Negócios, administração e direito', '04'),
    'LOGÃ\x8dSTICA': ('Negócios, administração e direito', '04'),
    'HOTELARIA HOSPITALAR': ('Saúde e bem-estar', '09'),
    'DESENVOLVIMENTO PARA WEB': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'FORMAÇÃO PEDAGÓGICA DE DOCENTES PARA A EDUCAÇÃO BÁSICA E PROFISSIONAL': ('Educação', '01'),
    'GESTÃO EMPREENDEDORA': ('Negócios, administração e direito', '04'),
    'PUBLICIDADE': ('Ciências sociais, comunicação e informação', '03'),
    'QUIROPRAXIA': ('Saúde e bem-estar', '09'),
    'OFTÁLMICA': ('Saúde e bem-estar', '09'),
    'CIÊNCIA DE DADOS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'TEATRO': ('Artes e humanidades', '02'),
    'ESTATÍSTICA': ('Ciências naturais, matemática e estatística', '05'),
    'RÁDIO, TV E INTERNET': ('Ciências sociais, comunicação e informação', '03'),
    'SECRETARIADO EXECUTIVO TRILÍNGUE': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO SOCIAL - RADIALISMO (RÁDIO E TV)': ('Ciências sociais, comunicação e informação', '03'),
    'SISTEMAS BIOMÉDICOS': ('Engenharia, produção e construção', '07'),
    'BIG DATA E INTELIGÊNCIA ANALÍTICA': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'LETRAS - FRANCÊS': ('Educação', '01'),
    'COMUNICAÇÃO EM CRIAÇÃO E DESENVOLVIMENTO DE WEB SITES E DESIGN': ('Ciências sociais, comunicação e informação', '03'),
    'SEGURANÇA NO TRÂNSITO': ('Serviços', '10'),
    'ELETRÔNICA INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'GESTÃO MERCADOLÓGICA': ('Negócios, administração e direito', '04'),
    'ENGENHARIA FLORESTAL': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'INTELIGÊNCIA ARTIFICIAL': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'CIÊNCIAS DA ACUPUNTURA': ('Saúde e bem-estar', '09'),
    'ENGENHARIA CARTOGRÁFICA E DE AGRIMENSURA': ('Engenharia, produção e construção', '07'),
    'MUSEOLOGIA': ('Ciências sociais, comunicação e informação', '03'),
    'TURISMO RECEPTIVO': ('Serviços', '10'),
    'COMUNICAÇÃO DIGITAL': ('Ciências sociais, comunicação e informação', '03'),
    'SERVIÇOS JUDICIAIS': ('Negócios, administração e direito', '04'),
    'DESENHO INDUSTRIAL': ('Artes e humanidades', '02'),
    'MÚSICA - MÚSICA POPULAR BRASILEIRA': ('Artes e humanidades', '02'),
    'VITICULTURA E ENOLOGIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'COMPUTAÇÃO EM NUVEM': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'SERVIÇOS JURÍDICOS': ('Negócios, administração e direito', '04'),
    'SERVIÇOS NOTARIAIS E REGISTRAIS': ('Negócios, administração e direito', '04'),
    'GESTÃO DO AGRONEGÓCIO': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'E-COMMERCE': ('Negócios, administração e direito', '04'),
    'SECRETARIADO EXECUTIVO BILINGUE - PORTUGUÊS/INGLÊS': ('Negócios, administração e direito', '04'),
    'GESTÃO DE TURISMO RECEPTIVO': ('Serviços', '10'),
    'FORMAÇÃO DE DOCENTES PARA A EDUCAÇÃO BÁSICA': ('Educação', '01'),
    'PRODUÇÃO SUCROALCOOLEIRA': ('Engenharia, produção e construção', '07'),
    'AGROINDUSTRIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'COMUNICAÇÃO E ILUSTRAÇÃO DIGITAL': ('Artes e humanidades', '02'),
    'CIÊNCIAS DA RELIGIÃO': ('Artes e humanidades', '02'),
    'PROCESSOS METALÚRGICOS': ('Engenharia, produção e construção', '07'),
    'CONTROLE DE OBRAS': ('Engenharia, produção e construção', '07'),
    'FABRICAÇÃO MECÂNICA': ('Engenharia, produção e construção', '07'),
    'GERONTOLOGIA': ('Saúde e bem-estar', '09'),
    'COSMÉTICOS': ('Engenharia, produção e construção', '07'),
    'PAISAGISMO': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'ENGENHARIA DE PESCA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'SUCROALCOOLEIRA': ('Engenharia, produção e construção', '07'),
    'SISTEMAS DE TELECOMUNICAÇÕES': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA AUTOMOTIVA': ('Engenharia, produção e construção', '07'),
    'DESENVOLVIMENTO E GESTÃO DE STARTUPS': ('Negócios, administração e direito', '04'),
    'GESTÃO DE STARTUP E NOVOS NEGÓCIOS': ('Negócios, administração e direito', '04'),
    'TECNOLOGIA E DESIGN DE NEGÓCIOS': ('Negócios, administração e direito', '04'),
    'INTERDISCIPLINAR EM CIÊNCIA E TECNOLOGIA': ('Ciências naturais, matemática e estatística', '05'),
    # --- NOMES NÃO ENCONTRADOS NO GABARITO ---
    'GESTÃO DE ENERGIAS': ('VERIFICAR MANUALMENTE', 'XX'), 

    # adicionado apos executar modulo3-> verificacao cursos nan, entao refeito todos csv do modulo 2 em diante( poucos refeitos)
    'ENGENHARIA DE ENERGIAS': ('Engenharia, produção e construção', '07'),
    }

    if not arquivo_entrada.exists():
        print(f"[!] ERRO: Arquivo base não encontrado: {arquivo_entrada}")
        return

    print(f"[*] Lendo arquivo unificado: {arquivo_entrada.name}...")
    df_principal = pd.read_parquet(str(arquivo_entrada))

    # =========================================================================
    # ETAPA A: CRIAR DATASET AUXILIAR (MAPEAMENTO VÁLIDO)
    # =========================================================================
    print("\n[*] ETAPA A: Extraindo dicionário de cursos válidos...")
    colunas_para_manter = [col_chave, col_val_nome, col_val_cod]
    
    df_mapa = df_principal[colunas_para_manter].copy()
    df_mapa_sem_nans = df_mapa.dropna(subset=[col_val_nome])
    df_auxiliar = df_mapa_sem_nans.drop_duplicates(subset=[col_chave]).reset_index(drop=True)

    caminho_auxiliar = pasta_data_03_temporarios / 'dataset_auxiliar_curso_area_inscritos.csv'
    df_auxiliar.to_csv(str(caminho_auxiliar), index=False, encoding='utf-8')
    print(f"  -> Salvo: {caminho_auxiliar.name} ({len(df_auxiliar)} cursos únicos sem NaN)")

    # =========================================================================
    # ETAPA B: ANÁLISE DOS NaNs
    # =========================================================================
    print("\n[*] ETAPA B: Analisando cursos com CINE não mapeado (NaNs)...")
    mask_nans = df_principal[col_val_nome].isnull()
    nans_antes = mask_nans.sum()
    print(f"  -> Total de NaNs encontrados ANTES da correção: {nans_antes}")

    if nans_antes > 0:
        # Agrupa para descobrir quais nomes de curso estão gerando os NaNs
        df_nans = df_principal[mask_nans]
        contagem_nomes_nan = df_nans[col_chave].value_counts(dropna=False)
        
        caminho_analise = pasta_data_03_temporarios / 'analise_nomes_nan_inscricao.csv'
        contagem_nomes_nan.to_csv(str(caminho_analise), header=['contagem'])
        print(f"  -> Lista de cursos problemáticos salva em: {caminho_analise.name}")

        # =========================================================================
        # ETAPA C: CORREÇÃO MANUAL VETORIZADA (ALTA PERFORMANCE)
        # =========================================================================
        print("\n[*] ETAPA C: Aplicando correções do dicionário manual...")
        
        # Separa o dicionário em dois mapeamentos rápidos
        map_nome = {k: v[0] for k, v in mapa_correcao_manual.items() if v[0] != 'VERIFICAR MANUALMENTE'}
        
        # O SEGREDO AQUI: Transforma o '04' (texto) em 4.0 (float) para bater com o INEP
        map_cod  = {k: float(v[1]) for k, v in mapa_correcao_manual.items() if v[0] != 'VERIFICAR MANUALMENTE'}

        # Aplica a correção apenas nas linhas que são NaN, buscando no dicionário
        df_principal.loc[mask_nans, col_val_nome] = df_principal.loc[mask_nans, col_chave].map(map_nome).fillna(df_principal.loc[mask_nans, col_val_nome])
        df_principal.loc[mask_nans, col_val_cod]  = df_principal.loc[mask_nans, col_chave].map(map_cod).fillna(df_principal.loc[mask_nans, col_val_cod])

        # Força a tipagem de toda a coluna para numérico, assim o PyArrow salva perfeitamente
        df_principal[col_val_cod] = pd.to_numeric(df_principal[col_val_cod], errors='coerce')

        nans_depois = df_principal[col_val_nome].isnull().sum()
        print(f"  -> Total de NaNs DEPOIS da correção: {nans_depois}")
        print(f"  -> Total de NaNs resolvidos: {nans_antes - nans_depois}")

        if nans_depois > 0:
            print("  [AVISO] Ainda restaram NaNs. Verifique o arquivo de análise e atualize o mapa_correcao_manual.")

    else:
        print("  -> Nenhum NaN encontrado! Correção desnecessária.")

    # =========================================================================
    # ETAPA D: SALVAR O CHECKPOINT CORRIGIDO
    # =========================================================================
    print("\n[*] ETAPA D: Salvando o Dataset Unificado Corrigido...")
    
    # Salvamos com o sufixo _corrigido para não perder o original em caso de erro
    arquivo_saida = pasta_data_03_transform_fies / 'fies_inscritos_unificado_corrigido.parquet'
    df_principal.to_parquet(str(arquivo_saida), index=False)
    
    print(f"  -> Arquivo final salvo com sucesso em: {arquivo_saida.name}")
    print("\n" + "="*60)
    print("✅ LAYER 2 CONCLUÍDA")
    print("="*60)
    auditoria_pos_correcao_inscritos()







def tratar_nans_cine_ofertas():
    print("\n" + "="*60)
    print("🛠️ INICIANDO LAYER 2: TRATAMENTO DE NaNs CINE - OFERTAS")
    print("="*60)


    arquivo_entrada = pasta_data_03_transform_fies / 'fies_ofertas_unificado.parquet'
    
    # Nomes das colunas padronizadas na Transform Layer 1 para Ofertas
    col_chave = 'nome_curso_ofertas'
    col_val_nome = 'NO_CINE_AREA_GERAL'
    col_val_cod = 'CO_CINE_AREA_GERAL'

    # --- 2. MAPA DE CORREÇÃO MANUAL PARA OFERTAS ---
    mapa_correcao_manual_ofertas = {
    'ADMINISTRAÇÃO': ('Negócios, administração e direito', '04'),
    'EDUCAÇÃO FÍSICA': ('Educação', '01'), # Corrigido!
    'CIÊNCIAS CONTÁBEIS': ('Negócios, administração e direito', '04'),
    'ENGENHARIA DE PRODUÇÃO': ('Engenharia, produção e construção', '07'),
    'PEDAGOGIA': ('Educação', '01'),
    'LOGÍSTICA': ('Negócios, administração e direito', '04'),
    'GESTÃO DE RECURSOS HUMANOS': ('Negócios, administração e direito', '04'),
    'ENGENHARIA CIVIL': ('Engenharia, produção e construção', '07'),
    'DIREITO': ('Negócios, administração e direito', '04'),
    'SERVIÇO SOCIAL': ('Saúde e bem-estar', '09'),
    'NUTRIÇÃO': ('Saúde e bem-estar', '09'),
    'FARMÁCIA': ('Saúde e bem-estar', '09'),
    'BIOMEDICINA': ('Saúde e bem-estar', '09'),
    'ARQUITETURA E URBANISMO': ('Engenharia, produção e construção', '07'),
    'MARKETING': ('Negócios, administração e direito', '04'),
    'FISIOTERAPIA': ('Saúde e bem-estar', '09'),
    'CIÊNCIAS BIOLÓGICAS': ('Ciências naturais, matemática e estatística', '05'),
    'ENGENHARIA ELÉTRICA': ('Engenharia, produção e construção', '07'),
    'ENFERMAGEM': ('Saúde e bem-estar', '09'),
    'REDES DE COMPUTADORES': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'ANÁLISE E DESENVOLVIMENTO DE SISTEMAS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'GESTÃO FINANCEIRA': ('Negócios, administração e direito', '04'),
    'GESTÃO COMERCIAL': ('Negócios, administração e direito', '04'),
    'SEGURANÇA NO TRABALHO': ('Engenharia, produção e construção', '07'),
    'SISTEMAS DE INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'PSICOLOGIA': ('Ciências sociais, comunicação e informação', '03'),
    'ESTÉTICA E COSMÉTICA': ('Serviços', '10'),
    'GESTÃO HOSPITALAR': ('Negócios, administração e direito', '04'),
    'ENGENHARIA MECÂNICA': ('Engenharia, produção e construção', '07'),
    'PROCESSOS GERENCIAIS': ('Negócios, administração e direito', '04'),
    'LETRAS - PORTUGUÊS E INGLÊS': ('Educação', '01'),
    'DESIGN DE INTERIORES': ('Artes e humanidades', '02'),
    'TURISMO': ('Serviços', '10'),
    'COMÉRCIO EXTERIOR': ('Negócios, administração e direito', '04'),
    'GASTRONOMIA': ('Serviços', '10'),
    'ENGENHARIA DE COMPUTAÇÃO': ('Engenharia, produção e construção', '07'),
    'LETRAS - LÍNGUA PORTUGUESA': ('Educação', '01'),
    'JORNALISMO': ('Ciências sociais, comunicação e informação', '03'),
    'GESTÃO PÚBLICA': ('Negócios, administração e direito', '04'),
    'ENGENHARIA AMBIENTAL E SANITÁRIA': ('Engenharia, produção e construção', '07'),
    'HISTÓRIA': ('Educação', '01'),
    'GESTÃO DA TECNOLOGIA DA INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'DESIGN GRÁFICO': ('Artes e humanidades', '02'),
    'ODONTOLOGIA': ('Saúde e bem-estar', '09'),
    'ENGENHARIA DE CONTROLE E AUTOMAÇÃO': ('Engenharia, produção e construção', '07'),
    'GESTÃO AMBIENTAL': ('Engenharia, produção e construção', '07'), # Ofertas Aux tem Engenharia
    'CIÊNCIA DA COMPUTAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'PUBLICIDADE E PROPAGANDA': ('Ciências sociais, comunicação e informação', '03'),
    'GESTÃO DE SEGURANÇA PRIVADA': ('Serviços', '10'),
    'SISTEMAS PARA INTERNET': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'ENGENHARIA QUÍMICA': ('Engenharia, produção e construção', '07'),
    'CONSTRUÇÃO DE EDIFÍCIOS': ('Engenharia, produção e construção', '07'),
    'RADIOLOGIA': ('Saúde e bem-estar', '09'),
    'GESTÃO DA QUALIDADE': ('Negócios, administração e direito', '04'),
    'GEOGRAFIA': ('Educação', '01'),
    'SECRETARIADO EXECUTIVO': ('Negócios, administração e direito', '04'),
    'MATEMÁTICA': ('Educação', '01'),
    'MEDICINA VETERINÁRIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'ENGENHARIA AGRONÔMICA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'RELAÇÕES INTERNACIONAIS': ('Ciências sociais, comunicação e informação', '03'),
    'JOGOS DIGITAIS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'DESIGN DE MODA': ('Artes e humanidades', '02'),
    'COMUNICAÇÃO SOCIAL - PUBLICIDADE E PROPAGANDA': ('Ciências sociais, comunicação e informação', '03'),
    'ENGENHARIA AMBIENTAL': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE ALIMENTOS': ('Engenharia, produção e construção', '07'),
    'CIÊNCIAS ECONÔMICAS': ('Ciências sociais, comunicação e informação', '03'),
    'PRODUÇÃO AUDIOVISUAL': ('Artes e humanidades', '02'),
    'LETRAS - PORTUGUÊS': ('Educação', '01'),
    'SEGURANÇA DA INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'DEFESA CIBERNÉTICA': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'QUÍMICA': ('Educação', '01'), # Auxiliar Ofertas só tem Educação
    'SISTEMA DE INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'ENGENHARIA MECATRÔNICA': ('Engenharia, produção e construção', '07'),
    'TERAPIA OCUPACIONAL': ('Saúde e bem-estar', '09'),
    'COMUNICAÇÃO SOCIAL - JORNALISMO': ('Ciências sociais, comunicação e informação', '03'),
    'AGRONOMIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'NEGÓCIOS IMOBILIÁRIOS': ('Negócios, administração e direito', '04'),
    'ENGENHARIA DE ENERGIA': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA AGRÍCOLA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'FOTOGRAFIA': ('Artes e humanidades', '02'),
    'SECRETARIADO': ('Negócios, administração e direito', '04'),
    'ARTES VISUAIS': ('Artes e humanidades', '02'),
    'ENGENHARIA DE TELECOMUNICAÇÕES': ('Engenharia, produção e construção', '07'),
    'LETRAS - ESPANHOL': ('Educação', '01'),
    'PRODUÇÃO MULTIMÍDIA': ('Artes e humanidades', '02'),
    'CIÊNCIAS SOCIAIS': ('Ciências sociais, comunicação e informação', '03'), # Auxiliar Ofertas tem Ciencias Sociais
    'LETRAS - INGLÊS': ('Educação', '01'),
    'RELAÇÕES PÚBLICAS': ('Ciências sociais, comunicação e informação', '03'),
    'FONOAUDIOLOGIA': ('Saúde e bem-estar', '09'),
    'COMUNICAÇÃO SOCIAL': ('Ciências sociais, comunicação e informação', '03'),
    'GESTÃO DE TURISMO': ('Serviços', '10'),
    'PETRÓLEO E GÁS': ('Engenharia, produção e construção', '07'),
    'EVENTOS': ('Serviços', '10'),
    'AUTOMAÇÃO INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'LETRAS': ('Educação', '01'),
    'ENGENHARIA DE PETRÓLEO': ('Engenharia, produção e construção', '07'),
    'HOTELARIA': ('Serviços', '10'),
    'ENGENHARIA DA COMPUTAÇÃO': ('Engenharia, produção e construção', '07'),
    'FÍSICA': ('Educação', '01'), # Auxiliar Ofertas só tem Educação
    'TEOLOGIA': ('Artes e humanidades', '02'),
    'PSICOPEDAGOGIA': ('Educação', '01'),
    'GESTÃO DA PRODUÇÃO INDUSTRIAL': ('Negócios, administração e direito', '04'),
    'CIÊNCIA POLÍTICA': ('Ciências sociais, comunicação e informação', '03'),
    'SERVIÇOS PENAIS': ('Serviços', '10'),
    'GESTÃO PORTUÁRIA': ('Serviços', '10'),
    'BANCO DE DADOS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'LETRAS - PORTUGUÊS E ESPANHOL': ('Educação', '01'),
    'DESIGN': ('Artes e humanidades', '02'),
    'DESIGN DE PRODUTO': ('Artes e humanidades', '02'),
    'GEOPROCESSAMENTO': ('Engenharia, produção e construção', '07'),
    'ADMINISTRAÇÃO PÚBLICA': ('Negócios, administração e direito', '04'),
    'FORMAÇÃO PEDAGÓGICA PARA PORTADORES DE ENSINO SUPERIOR': ('Educação', '01'),
    'MÚSICA': ('Artes e humanidades', '02'),
    'FILOSOFIA': ('Artes e humanidades', '02'),
    'MANUTENÇÃO INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'GESTÃO DE EMPREENDIMENTOS ESPORTIVOS': ('Serviços', '10'),
    'ENGENHARIA ELETRÔNICA': ('Engenharia, produção e construção', '07'),
    'MODA': ('Artes e humanidades', '02'),
    'ENGENHARIA DE PETRÓLEO E GÁS': ('Engenharia, produção e construção', '07'),
    'SECRETARIADO EXECUTIVO TRILINGUE': ('Negócios, administração e direito', '04'),
    'GESTÃO DE SERVIÇOS JURÍDICOS, NOTARIAIS E DE REGISTRO': ('Negócios, administração e direito', '04'),
    'CIÊNCIA ECONÔMICA': ('Ciências sociais, comunicação e informação', '03'),
    'SISTEMAS ELÉTRICOS': ('Engenharia, produção e construção', '07'),
    'COMUNICAÇÃO SOCIAL - RADIO E TELEVISÃO': ('Ciências sociais, comunicação e informação', '03'),
    'ENGENHARIA BIOMÉDICA': ('Engenharia, produção e construção', '07'),
    'MARKETING DIGITAL': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO INSTITUCIONAL': ('Ciências sociais, comunicação e informação', '03'),
    'COMUNICAÇÃO EMPRESARIAL': ('Ciências sociais, comunicação e informação', '03'),
    'CIÊNCIA DE DADOS E MACHINE LEARNING': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'SEGURANÇA PÚBLICA': ('Serviços', '10'),
    'GESTÃO DE PRODUÇÃO INDUSTRIAL': ('Negócios, administração e direito', '04'),
    'ENERGIAS RENOVÁVEIS': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE BIOPROCESSOS E BIOTECNologia': ('Engenharia, produção e construção', '07'),
    'ZOOTECNIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'BIOCOMBUSTÍVEIS': ('Engenharia, produção e construção', '07'),
    'PROCESSOS QUÍMICOS': ('Engenharia, produção e construção', '07'),
    'GESTÃO DE RECURSOS HÍDRICOS': ('Engenharia, produção e construção', '07'),
    'TRANSPORTE TERRESTRE': ('Serviços', '10'),
    'GESTÃO DESPORTIVA E DE LAZER': ('Serviços', '10'),
    'ELETROTÉCNICA INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'ESTÉTICA': ('Serviços', '10'),
    'GERENCIAMENTO DE REDES DE COMPUTADORES': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'AGRIMENSURA': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE SOFTWARE': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'GESTÃO DE COOPERATIVAS': ('Negócios, administração e direito', '04'),
    'ENGENHARIA DE MATERIAIS': ('Engenharia, produção e construção', '07'),
    'MECATRÔNICA INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'AGROPECUÁRIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'PRODUÇÃO GRÁFICA DIGITAL': ('Artes e humanidades', '02'),
    'BIBLIOTECONOMia': ('Ciências sociais, comunicação e informação', '03'), # Mantendo como no auxiliar
    'DANÇA': ('Artes e humanidades', '02'),
    'SEGURANÇA PRIVADA': ('Serviços', '10'),
    'TECNÓLOGO EM METALURGIA': ('Engenharia, produção e construção', '07'),
    'POLÍTICAS PÚBLICAS': ('Ciências sociais, comunicação e informação', '03'),
    'SERVIÇOS JURÍDICOS, CARTORÁRIOS E NOTARIAIS': ('Negócios, administração e direito', '04'),
    'PROPAGANDA E MARKETING': ('Negócios, administração e direito', '04'),
    'AGRONEGÓCIO': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'CONSTRUÇÃO NAVAL': ('Engenharia, produção e construção', '07'),
    'LETRAS - LÍNGUA PORTUGUESA E LITERATURAS DE LÍNGUA PORTUGUESA': ('Educação', '01'),
    'GESTÃO DE SERVIÇOS JURÍDICOS E NOTARIAIS': ('Negócios, administração e direito', '04'),
    'PAPEL E CELULOSE': ('Engenharia, produção e construção', '07'),
    'COMUNICAÇÃO PARA WEB': ('Ciências sociais, comunicação e informação', '03'),
    'PROCESSOS ESCOLARES': ('Educação', '01'),
    'EDUCAÇÃO ARTÍSTICA': ('Educação', '01'),
    'PRODUÇÃO CULTURAL': ('Artes e humanidades', '02'),
    'ACUPUNTURA': ('Saúde e bem-estar', '09'),
    'COMUNICAÇÃO SOCIAL\xa0 - RADIALISMO': ('Ciências sociais, comunicação e informação', '03'),
    'DESIGN DE ANIMAÇÃO': ('Artes e humanidades', '02'),
    'ARTES CÊNICAS': ('Artes e humanidades', '02'),
    'PILOTAGEM PROFISSIONAL DE AERONAVES': ('Serviços', '10'),
    'LOGÃ\x8dSTICA': ('Negócios, administração e direito', '04'), # Corrigindo encoding
    'ENGENHARIA DE BIOPROCESSOS E BIOTECNOLOGIA': ('Engenharia, produção e construção', '07'),
    'SECRETARIADO EXECUTIVO TRILÍNGUE': ('Negócios, administração e direito', '04'),
    'DESENVOLVIMENTO PARA WEB': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'VISAGISMO E TERAPIAS CAPILARES': ('Serviços', '10'),
    'PRODUÇÃO PUBLICITÁRIA': ('Ciências sociais, comunicação e informação', '03'),
    'FORMAÇÃO PEDAGÓGICA DE DOCENTES PARA A EDUCAÇÃO BÁSICA E PROFISSIONAL': ('Educação', '01'),
    'GESTÃO DE NEGÓCIOS E INOVAÇÃO': ('Negócios, administração e direito', '04'),
    'GESTÃO DE NEGÓCIOS NO VAREJO': ('Negócios, administração e direito', '04'),
    'COMUNICAÇÃO SOCIAL - RELAÇÕES PÚBLICAS': ('Ciências sociais, comunicação e informação', '03'),
    'GESTÃO EMPREENDEDORA': ('Negócios, administração e direito', '04'),
    'CINEMA E AUDIOVISUAL': ('Artes e humanidades', '02'),
    'ENGENHARIA AUTOMOTIVA': ('Engenharia, produção e construção', '07'),
    'INTELIGÊNCIA ARTIFICIAL': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'CIÊNCIAS DA ACUPUNTURA': ('Saúde e bem-estar', '09'),
    'OFTÁLMICA': ('Saúde e bem-estar', '09'),
    'ESTATÍSTICA': ('Ciências naturais, matemática e estatística', '05'),
    'RÁDIO, TV E INTERNET': ('Ciências sociais, comunicação e informação', '03'),
    'COMUNICAÇÃO EM CRIAÇÃO E DESENVOLVIMENTO DE WEB SITES E DESIGN': ('Ciências sociais, comunicação e informação', '03'),
    'TEATRO': ('Artes e humanidades', '02'),
    'GESTÃO MERCADOLÓGICA': ('Negócios, administração e direito', '04'),
    'ELETRÔNICA INDUSTRIAL': ('Engenharia, produção e construção', '07'),
    'BIG DATA E INTELIGÊNCIA ANALÍTICA': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'ENGENHARIA FLORESTAL': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'COMUNICAÇÃO SOCIAL - RADIALISMO (RÁDIO E TV)': ('Ciências sociais, comunicação e informação', '03'),
    'LETRAS - FRANCÊS': ('Educação', '01'),
    'SERVIÇOS JUDICIAIS': ('Negócios, administração e direito', '04'),
    'TURISMO RECEPTIVO': ('Serviços', '10'),
    'MEDIAÇÃO': ('Ciências sociais, comunicação e informação', '03'),
    'GESTÃO DE RESÍDUOS SÓLIDOS': ('Engenharia, produção e construção', '07'),
    'FABRICAÇÃO MECÂNICA': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA AEROESPACIAL': ('Engenharia, produção e construção', '07'),
    'DESENHO INDUSTRIAL': ('Artes e humanidades', '02'),
    'COMUNICAÇÃO DIGITAL': ('Ciências sociais, comunicação e informação', '03'),
    'QUIROPRAXIA': ('Saúde e bem-estar', '09'),
    'PODOLOGIA': ('Saúde e bem-estar', '09'),
    'MUSEOLOGIA': ('Ciências sociais, comunicação e informação', '03'),
    'VITICULTURA E ENOLOGIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'EDUCAÇÃO ESPECIAL': ('Educação', '01'),
    'ESTÉTICA E COSMETOLOGIA': ('Serviços', '10'),
    'ENGENHARIA CARTOGRÁFICA E DE AGRIMENSURA': ('Engenharia, produção e construção', '07'),
    'SERVIÇOS JURÍDICOS': ('Negócios, administração e direito', '04'),
    'SECRETARIADO EXECUTIVO BILINGUE - PORTUGUÊS/INGLÊS': ('Negócios, administração e direito', '04'),
    'E-COMMERCE': ('Negócios, administração e direito', '04'),
    'SERVIÇOS NOTARIAIS E REGISTRAIS': ('Negócios, administração e direito', '04'),
    'GESTÃO DE TURISMO RECEPTIVO': ('Serviços', '10'),
    'PROCESSOS METALÚRGICOS': ('Engenharia, produção e construção', '07'),
    'AGROINDUSTRIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'FORMAÇÃO DE DOCENTES PARA A EDUCAÇÃO BÁSICA': ('Educação', '01'),
    'MÚSICA - MÚSICA POPULAR BRASILEIRA': ('Artes e humanidades', '02'),
    'COMPUTAÇÃO EM NUVEM': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'PRODUÇÃO SUCROALCOOLEIRA': ('Engenharia, produção e construção', '07'),
    'CIÊNCIAS DA RELIGIÃO': ('Artes e humanidades', '02'),
    'COMUNICAÇÃO E ILUSTRAÇÃO DIGITAL': ('Artes e humanidades', '02'),
    'SISTEMAS BIOMÉDICOS': ('Engenharia, produção e construção', '07'),
    'SEGURANÇA NO TRÂNSITO': ('Serviços', '10'),
    'PAISAGISMO': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'GESTÃO DO AGRONEGÓCIO': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'CIÊNCIA DE DADOS': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'HOTELARIA HOSPITALAR': ('Saúde e bem-estar', '09'),
    'PUBLICIDADE': ('Ciências sociais, comunicação e informação', '03'),
    'CONTROLE DE OBRAS': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE PESCA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'GESTÃO DE STARTUP E NOVOS NEGÓCIOS': ('Negócios, administração e direito', '04'),
    'GERONTOLOGIA': ('Saúde e bem-estar', '09'),
    'COSMÉTICOS': ('Engenharia, produção e construção', '07'),
    'ENGENHARIA DE SISTEMAS': ('Engenharia, produção e construção', '07'),
    'GESTÃO INTEGRADA DE AGRONEGÓCIOS': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'LETRAS - LIBRAS': ('Educação', '01'),
    'SUCROALCOOLEIRA': ('Engenharia, produção e construção', '07'),
    'SISTEMAS DE TELECOMUNICAÇÕES': ('Engenharia, produção e construção', '07'),
    '"RÁDIO, TV E INTERNET"': ('Ciências sociais, comunicação e informação', '03'),
    'BIBLIOTECONOMIA': ('Ciências sociais, comunicação e informação', '03'), # Assumindo que "BIBLIOTECONOMia" era typo
    # --- NOMES AINDA NÃO ENCONTRADOS OU AMBÍGUOS ---
    'GESTÃO DE HOTELARIA': ('VERIFICAR MANUALMENTE', 'XX'), # Não encontrado no auxiliar
    'GESTÃO FINANCEIRA PÚBLICA': ('VERIFICAR MANUALMENTE', 'XX'), # Não encontrado no auxiliar
    'ENGENHARIA DE SEGURANÇA NO TRABALHO': ('VERIFICAR MANUALMENTE', 'XX'), # Não encontrado no auxiliar
    'COMUNICAÇÃO SOCIAL - CINEMA E VÍDEO': ('VERIFICAR MANUALMENTE', 'XX'), # Não encontrado no auxiliar
    'PRODUÇÃO DE GRAFOS': ('VERIFICAR MANUALMENTE', 'XX'), # Não encontrado no auxiliar





    # adicionado apos executar modulo3-> verificacao cursos nan, entao refeito todos csv do modulo 2 em diante( poucos refeitos)

    # --- NOVOS CURSOS (Da sua lista de 59 NaNs) ---
    'CIÊNCIAS DA COMPUTAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'ENGENHARIA': ('Engenharia, produção e construção', '07'), # Geralmente o campo ENGENHARIA sem detalhe cai em 07
    'BIOLOGIA': ('Ciências naturais, matemática e estatística', '05'), # Assumindo Bacharelado/Licenciatura
    'LETRAS PORTUGUÊS E INGLÊS': ('Educação', '01'),
    'ENGENHARIA DE ENERGIAS': ('Engenharia, produção e construção', '07'),
    'PRODUÇÃO DE GRÃOS': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    'INTERDISCIPLINAR EM CIÊNCIA E TECNOLOGIA': ('Ciências naturais, matemática e estatística', '05'), # Cursos Interdisciplinares caem em 05 ou 02, escolhemos 05.
    'TECNOLOGIA E DESIGN DE NEGÓCIOS': ('Negócios, administração e direito', '04'),
    'CIÊNCIA DA INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
    'NORMAL SUPERIOR': ('Educação', '01'),
    'COMUNICAÇÃO SOCIAL - CINEMA E AUDIOVISUAL': ('Artes e humanidades', '02'),
    'COMUNICAÇÃO SOCIAL COM HABILITAÇÃO EM PUBLICID...': ('Ciências sociais, comunicação e informação', '03'), # Publicidade
    'MULTIMÍDIA': ('Artes e humanidades', '02'),
    'LETRAS - LINGUAGEM AUDIOVISUAL': ('Artes e humanidades', '02'),
    'ESTILISMO': ('Artes e humanidades', '02'), # Design de Moda/Artes
    'GESTÃO DE INVESTIMENTOS': ('Negócios, administração e direito', '04'),
    'DESENVOLVIMENTO E GESTÃO DE STARTUPS': ('Negócios, administração e direito', '04'),
    'EMPREENDEDORISMO': ('Negócios, administração e direito', '04'),
    'COACHING': ('Serviços', '10'), # Serviços/Negócios
    'PRODUÇÃO CERVEJEIRA': ('Engenharia, produção e construção', '07'), # Processos Industriais/Eng. Química
    # --- CURSOS QUE JÁ EXISTIAM NO SEU MAPA, MAS COM OUTROS NOMES/VÍRGULAS/SPACING ---
    'CIÊNCIA DA COMPUTAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'), # Mesmo que CIÊNCIAS DA COMPUTAÇÃO
    # --- NOMES AMBÍGUOS OU INCOMPLETOS JÁ LISTADOS ---
    # (Mantendo apenas o que não estava no seu mapa anterior, que não enviei)
    
    # --- CURSOS COMUNS (Mantendo os que você já tinha no original como base) ---
    'ADMINISTRAÇÃO': ('Negócios, administração e direito', '04'),
    'EDUCAÇÃO FÍSICA': ('Saúde e bem-estar', '09'),
    'CIÊNCIAS CONTÁBEIS': ('Negócios, administração e direito', '04'),
    'PEDAGOGIA': ('Educação', '01'),
    'ENGENHARIA CIVIL': ('Engenharia, produção e construção', '07'),
    'DIREITO': ('Negócios, administração e direito', '04'),
    'FARMÁCIA': ('Saúde e bem-estar', '09'),
    'PSICOLOGIA': ('Ciências sociais, comunicação e informação', '03'),
    'JORNALISMO': ('Ciências sociais, comunicação e informação', '03'),
    'ODONTOLOGIA': ('Saúde e bem-estar', '09'),
    'TERAPIA OCUPACIONAL': ('Saúde e bem-estar', '09'),
    'AGRONOMIA': ('Agricultura, silvicultura, pesca e veterinária', '08'),
    # (Adicione aqui o restante dos seus mapeamentos já feitos para garantir que eles não sumam)
    
    # --- NOMES QUE PRECISAM DE LIMPEZA EXTRA (Se vierem sujos) ---
    'LOGÍSTICA': ('Negócios, administração e direito', '04'), # Exemplo de curso que precisa ser adicionado
    'SERVIÇO SOCIAL': ('Saúde e bem-estar', '09'), # Exemplo de curso que precisa ser adicionado
    
    # --- OUTROS CURSOS QUE VOCÊ JÁ TINHA MAIS OTIMIZADOS ---
    # Copie e cole o restante dos seus 100+ mapeamentos já prontos aqui!
    }

    if not arquivo_entrada.exists():
        print(f"[!] ERRO: Arquivo base de ofertas não encontrado: {arquivo_entrada}")
        return

    print(f"[*] Lendo arquivo unificado de ofertas: {arquivo_entrada.name}...")
    df_principal = pd.read_parquet(str(arquivo_entrada))

    # =========================================================================
    # ETAPA A: CRIAR DATASET AUXILIAR (MAPEAMENTO VÁLIDO)
    # =========================================================================
    print("\n[*] ETAPA A: Extraindo dicionário de cursos válidos (Ofertas)...")
    colunas_para_manter = [col_chave, col_val_nome, col_val_cod]
    
    df_mapa = df_principal[colunas_para_manter].copy()
    df_mapa_sem_nans = df_mapa.dropna(subset=[col_val_nome])
    df_auxiliar = df_mapa_sem_nans.drop_duplicates(subset=[col_chave]).reset_index(drop=True)

    caminho_auxiliar = pasta_data_03_temporarios / 'dataset_auxiliar_curso_area_ofertas.csv'
    df_auxiliar.to_csv(str(caminho_auxiliar), index=False, encoding='utf-8')
    print(f"  -> Salvo: {caminho_auxiliar.name} ({len(df_auxiliar)} cursos únicos)")

    # =========================================================================
    # ETAPA B: ANÁLISE DOS NaNs
    # =========================================================================
    print("\n[*] ETAPA B: Analisando NaNs em Ofertas...")
    mask_nans = df_principal[col_val_nome].isnull()
    nans_antes = mask_nans.sum()
    print(f"  -> Total de NaNs encontrados ANTES da correção: {nans_antes}")

    if nans_antes > 0:
        df_nans = df_principal[mask_nans]
        contagem_nomes_nan = df_nans[col_chave].value_counts(dropna=False)
        
        caminho_analise = pasta_data_03_temporarios / 'analise_nomes_nan_ofertas.csv'
        contagem_nomes_nan.to_csv(str(caminho_analise), header=['contagem'])
        print(f"  -> Lista de cursos problemáticos salva em: {caminho_analise.name}")

        # =========================================================================
        # ETAPA C: CORREÇÃO MANUAL VETORIZADA
        # =========================================================================
        print("\n[*] ETAPA C: Aplicando correções manuais (Vetorizado)...")
        
        # Mapeamentos rápidos com conversão de texto para float no código
        map_nome = {k: v[0] for k, v in mapa_correcao_manual_ofertas.items() if v[0] != 'VERIFICAR MANUALMENTE'}
        map_cod  = {k: float(v[1]) for k, v in mapa_correcao_manual_ofertas.items() if v[0] != 'VERIFICAR MANUALMENTE'}

        # Imputação cirúrgica: apenas onde é NaN
        df_principal.loc[mask_nans, col_val_nome] = df_principal.loc[mask_nans, col_chave].map(map_nome).fillna(df_principal.loc[mask_nans, col_val_nome])
        df_principal.loc[mask_nans, col_val_cod]  = df_principal.loc[mask_nans, col_chave].map(map_cod).fillna(df_principal.loc[mask_nans, col_val_cod])

        # Padroniza a coluna para float64 pro Parquet
        df_principal[col_val_cod] = pd.to_numeric(df_principal[col_val_cod], errors='coerce')

        nans_depois = df_principal[col_val_nome].isnull().sum()
        print(f"  -> Total de NaNs DEPOIS da correção: {nans_depois}")
        print(f"  -> Total de NaNs resolvidos: {nans_antes - nans_depois}")
    else:
        print("  -> Nenhum NaN encontrado nas ofertas.")

    # =========================================================================
    # ETAPA D: SALVAR O CHECKPOINT CORRIGIDO
    # =========================================================================
    print("\n[*] ETAPA D: Salvando o Dataset de Ofertas Corrigido...")
    
    arquivo_saida = pasta_data_03_transform_fies / 'fies_ofertas_unificado_corrigido.parquet'
    # Garante a ordenação cronológica definida no projeto
    df_final = df_principal.sort_values(by=['ano_ofertas', 'semestre_ofertas'])
    df_final.to_parquet(str(arquivo_saida), index=False)
    
    print(f"  -> Arquivo final salvo com sucesso em: {arquivo_saida.name}")
    print("\n" + "="*60)
    print("✅ LAYER 2 CONCLUÍDA (OFERTAS)")
    print("="*60)
    auditoria_pos_correcao_ofertas()













def auditoria_pos_correcao_inscritos():
    print("\n" + "="*80)
    print("🎯 AUDITORIA PÓS-CORREÇÃO (LAYER 2): FIES INSCRITOS")
    print("="*80)

    arquivo_parquet = pasta_raiz_projeto / 'data' / '03_transform' / 'fies' / 'fies_inscritos_unificado_corrigido.parquet'
    coluna_analise = 'NO_CINE_AREA_GERAL'

    if not arquivo_parquet.exists():
        print(f"[!] ERRO: Arquivo corrigido não encontrado: {arquivo_parquet.name}")
        return

    # Fallback para display em ambientes interativos
    try:
        from IPython.display import display
    except ImportError:
        display = print

    # 2. Leitura e Análise
    df = pd.read_parquet(str(arquivo_parquet))
    
    if coluna_analise not in df.columns:
        print(f"  ERRO: A coluna '{coluna_analise}' não foi encontrada!")
        return

    # --- 3. CONTAGEM GLOBAL ---
    print(f"\n📊 Contagem Global de Valores para '{coluna_analise}':")
    contagem = df[coluna_analise].value_counts(dropna=False)
    print(contagem)

    # --- 4. RESUMO DE NaNs ---
    total_linhas = len(df)
    total_nans = df[coluna_analise].isnull().sum()
    percentual_nans = (total_nans / total_linhas) * 100

    print(f"\n{'-'*40}")
    print(f"✅ RESUMO DE INTEGRIDADE (FINAL)")
    print(f"{'-'*40}")
    print(f"Total de Inscritos Processados : {total_linhas}")
    print(f"Total de NaNs (Não Mapeados)    : {total_nans}")
    print(f"Taxa de Perda Residual          : {percentual_nans:.2f}%")
    print(f"{'='*80}\n")


def auditoria_pos_correcao_ofertas():
    print("\n" + "="*80)
    print("🎯 AUDITORIA PÓS-CORREÇÃO (LAYER 2): FIES OFERTAS")
    print("="*80)

    arquivo_parquet = pasta_raiz_projeto / 'data' / '03_transform' / 'fies' / 'fies_ofertas_unificado_corrigido.parquet'
    coluna_analise = 'NO_CINE_AREA_GERAL'

    if not arquivo_parquet.exists():
        print(f"[!] ERRO: Arquivo corrigido não encontrado: {arquivo_parquet.name}")
        return

    # Fallback para display em ambientes interativos (Jupyter)
    try:
        from IPython.display import display
    except ImportError:
        display = print

    # 2. Leitura do Parquet Corrigido
    df = pd.read_parquet(str(arquivo_parquet))
    
    if coluna_analise not in df.columns:
        print(f"  ERRO: A coluna '{coluna_analise}' não foi encontrada!")
        return

    # --- 3. CONTAGEM GLOBAL DE ÁREAS ---
    print(f"\n📊 Contagem Global de Valores para '{coluna_analise}':")
    contagem = df[coluna_analise].value_counts(dropna=False)
    print(contagem)

    # --- 4. RESUMO DE NaNs (Vagas sem CINE) ---
    total_linhas = len(df)
    total_nans = df[coluna_analise].isnull().sum()
    percentual_nans = (total_nans / total_linhas) * 100

    print(f"\n{'-'*40}")
    print(f"✅ RESUMO DE INTEGRIDADE: OFERTAS")
    print(f"{'-'*40}")
    print(f"Total de Ofertas Processadas   : {total_linhas}")
    print(f"Total de NaNs (Não Mapeados)   : {total_nans}")
    print(f"Taxa de Perda Residual         : {percentual_nans:.2f}%")
    print(f"{'='*80}\n")