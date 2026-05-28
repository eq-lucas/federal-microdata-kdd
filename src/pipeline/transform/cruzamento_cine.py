from pathlib import Path
import re
import unicodedata

import pandas as pd

from src.constants import (
    INTERIM_FIES_DIR,
    INTERIM_INEP_DIR,
    LOGS_DIR,
    TEMP_DIR,
)


ARQUIVO_INSCRICOES_UNIFICADAS = INTERIM_FIES_DIR / "fies_inscricoes_unificadas.parquet"
ARQUIVO_OFERTAS_UNIFICADAS = INTERIM_FIES_DIR / "fies_ofertas_unificadas.parquet"

ARQUIVO_INEP_HISTORICO = INTERIM_INEP_DIR / "inep_cursos_cine_historico_2016_2024.parquet"

ARQUIVO_INSCRICOES_COM_CINE = INTERIM_FIES_DIR / "fies_inscricoes_com_cine.parquet"
ARQUIVO_OFERTAS_COM_CINE = INTERIM_FIES_DIR / "fies_ofertas_com_cine.parquet"

RESUMO_CRUZAMENTO_CINE = LOGS_DIR / "transform_cruzamento_cine_resumo.csv"

COL_CURSO = "nome_curso"
COL_CINE_NOME = "nome_cine_area_geral"
COL_CINE_COD = "codigo_cine_area_geral"


COLUNAS_INEP_BASE = [
    "codigo_curso",
    "ano_censo",
    "nome_curso_inep",
    "codigo_cine_area_geral",
    "nome_cine_area_geral",
    "codigo_cine_area_especifica",
    "nome_cine_area_especifica",
    "codigo_cine_area_detalhada",
    "nome_cine_area_detalhada",
    "codigo_cine_rotulo",
    "nome_cine_rotulo",
    "arquivo_origem",
]

COLUNAS_INEP_RENOMEADAS = {
    "ano_censo": "inep_ano_censo",
    "arquivo_origem": "inep_arquivo_origem",
}


# Mapas manuais vindos do seu transform_layer_2.py antigo.
# Eles ficam aqui porque, neste repositório limpo, a correção dos CINE faltantes
# é parte da própria etapa de cruzamento CINE.
MAPA_CORRECAO_MANUAL_INSCRICOES = {'DIREITO': ('Negócios, administração e direito', '04'),
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
 'GESTÃO DE ENERGIAS': ('VERIFICAR MANUALMENTE', 'XX'),
 'ENGENHARIA DE ENERGIAS': ('Engenharia, produção e construção', '07')}

MAPA_CORRECAO_MANUAL_OFERTAS = {'ADMINISTRAÇÃO': ('Negócios, administração e direito', '04'),
 'EDUCAÇÃO FÍSICA': ('Saúde e bem-estar', '09'),
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
 'GESTÃO AMBIENTAL': ('Engenharia, produção e construção', '07'),
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
 'QUÍMICA': ('Educação', '01'),
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
 'CIÊNCIAS SOCIAIS': ('Ciências sociais, comunicação e informação', '03'),
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
 'FÍSICA': ('Educação', '01'),
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
 'BIBLIOTECONOMia': ('Ciências sociais, comunicação e informação', '03'),
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
 'LOGÃ\x8dSTICA': ('Negócios, administração e direito', '04'),
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
 'BIBLIOTECONOMIA': ('Ciências sociais, comunicação e informação', '03'),
 'GESTÃO DE HOTELARIA': ('VERIFICAR MANUALMENTE', 'XX'),
 'GESTÃO FINANCEIRA PÚBLICA': ('VERIFICAR MANUALMENTE', 'XX'),
 'ENGENHARIA DE SEGURANÇA NO TRABALHO': ('VERIFICAR MANUALMENTE', 'XX'),
 'COMUNICAÇÃO SOCIAL - CINEMA E VÍDEO': ('VERIFICAR MANUALMENTE', 'XX'),
 'PRODUÇÃO DE GRAFOS': ('VERIFICAR MANUALMENTE', 'XX'),
 'CIÊNCIAS DA COMPUTAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
 'ENGENHARIA': ('Engenharia, produção e construção', '07'),
 'BIOLOGIA': ('Ciências naturais, matemática e estatística', '05'),
 'LETRAS PORTUGUÊS E INGLÊS': ('Educação', '01'),
 'ENGENHARIA DE ENERGIAS': ('Engenharia, produção e construção', '07'),
 'PRODUÇÃO DE GRÃOS': ('Agricultura, silvicultura, pesca e veterinária', '08'),
 'INTERDISCIPLINAR EM CIÊNCIA E TECNOLOGIA': ('Ciências naturais, matemática e estatística', '05'),
 'TECNOLOGIA E DESIGN DE NEGÓCIOS': ('Negócios, administração e direito', '04'),
 'CIÊNCIA DA INFORMAÇÃO': ('Computação e Tecnologias da Informação e Comunicação (TIC)', '06'),
 'NORMAL SUPERIOR': ('Educação', '01'),
 'COMUNICAÇÃO SOCIAL - CINEMA E AUDIOVISUAL': ('Artes e humanidades', '02'),
 'COMUNICAÇÃO SOCIAL COM HABILITAÇÃO EM PUBLICID...': ('Ciências sociais, comunicação e informação', '03'),
 'MULTIMÍDIA': ('Artes e humanidades', '02'),
 'LETRAS - LINGUAGEM AUDIOVISUAL': ('Artes e humanidades', '02'),
 'ESTILISMO': ('Artes e humanidades', '02'),
 'GESTÃO DE INVESTIMENTOS': ('Negócios, administração e direito', '04'),
 'DESENVOLVIMENTO E GESTÃO DE STARTUPS': ('Negócios, administração e direito', '04'),
 'EMPREENDEDORISMO': ('Negócios, administração e direito', '04'),
 'COACHING': ('Serviços', '10'),
 'PRODUÇÃO CERVEJEIRA': ('Engenharia, produção e construção', '07')}


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "transform_cruzamento_cine.log"

    with log_path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(str(message) + "\n")

    print(message)


def normalizar_nome(valor) -> str | None:
    if pd.isna(valor):
        return None

    texto = str(valor).strip().upper()
    texto = texto.replace("\xa0", " ")
    texto = re.sub(r"\s+", " ", texto).strip()

    if texto in {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", "--"}:
        return None

    return texto


def normalizar_nome_sem_acento(valor) -> str | None:
    texto = normalizar_nome(valor)

    if texto is None:
        return None

    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def normalizar_codigo_cine(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()

    if texto in {"", "nan", "NaN", "None", "NONE", "NULL", "-", "--"}:
        return pd.NA

    try:
        numero = int(float(texto.replace(",", ".")))
        return str(numero).zfill(2)
    except Exception:
        return texto.upper()


def preparar_mapa_manual(mapa: dict) -> tuple[dict, dict, dict, dict]:
    mapa_nome_exato = {}
    mapa_cod_exato = {}
    mapa_nome_norm = {}
    mapa_cod_norm = {}

    for curso, valor in mapa.items():
        if not isinstance(valor, tuple) or len(valor) < 2:
            continue

        nome_area, codigo_area = valor[0], valor[1]

        if nome_area == "VERIFICAR MANUALMENTE" or codigo_area == "XX":
            continue

        curso_exato = normalizar_nome(curso)
        curso_norm = normalizar_nome_sem_acento(curso)

        if curso_exato is None:
            continue

        nome_area_padrao = normalizar_nome(nome_area)
        codigo_area_padrao = normalizar_codigo_cine(codigo_area)

        mapa_nome_exato[curso_exato] = nome_area_padrao
        mapa_cod_exato[curso_exato] = codigo_area_padrao

        if curso_norm is not None:
            mapa_nome_norm[curso_norm] = nome_area_padrao
            mapa_cod_norm[curso_norm] = codigo_area_padrao

    return mapa_nome_exato, mapa_cod_exato, mapa_nome_norm, mapa_cod_norm


def preparar_inep() -> pd.DataFrame:
    if not ARQUIVO_INEP_HISTORICO.exists():
        raise FileNotFoundError(f"Mestre INEP não encontrado: {ARQUIVO_INEP_HISTORICO}")

    inep = pd.read_parquet(ARQUIVO_INEP_HISTORICO)

    colunas_existentes = []
    for col in COLUNAS_INEP_BASE:
        if col in inep.columns and col not in colunas_existentes:
            colunas_existentes.append(col)

    inep = inep[colunas_existentes].copy()

    if "codigo_curso" not in inep.columns or "ano_censo" not in inep.columns:
        raise ValueError("Mestre INEP precisa conter codigo_curso e ano_censo.")

    inep["codigo_curso"] = pd.to_numeric(inep["codigo_curso"], errors="coerce").astype("Int64")
    inep["ano_censo"] = pd.to_numeric(inep["ano_censo"], errors="coerce").astype("Int64")

    inep = inep.dropna(subset=["codigo_curso", "ano_censo"]).copy()

    inep = inep.sort_values(
        by=["codigo_curso", "ano_censo"],
        ascending=[True, True],
        kind="mergesort",
    )

    inep = inep.drop_duplicates(
        subset=["codigo_curso", "ano_censo"],
        keep="last",
    ).reset_index(drop=True)

    inep = inep.rename(columns=COLUNAS_INEP_RENOMEADAS)

    return inep


def preparar_base_fies(path: Path, ano_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo FIES não encontrado: {path}")

    df = pd.read_parquet(path)

    if "codigo_curso" not in df.columns:
        raise ValueError(f"{path.name} não contém codigo_curso.")

    if ano_col not in df.columns:
        raise ValueError(f"{path.name} não contém {ano_col}.")

    df = df.copy()
    df["codigo_curso"] = pd.to_numeric(df["codigo_curso"], errors="coerce").astype("Int64")
    df[ano_col] = pd.to_numeric(df[ano_col], errors="coerce").astype("Int64")

    return df


def construir_referencia_cine(inep: pd.DataFrame, anos_fies: list[int]) -> pd.DataFrame:
    """
    Constrói referência CINE por codigo_curso e ano FIES.

    Regra:
    1. mesmo ano do FIES;
    2. se não existir, Censo passado mais recente;
    3. se não existir, Censo futuro mais próximo.
    """
    anos_fies = sorted({int(ano) for ano in anos_fies if pd.notna(ano)})

    if not anos_fies:
        raise ValueError("Não há anos FIES válidos para construir referência CINE.")

    inep = inep.copy()
    inep["codigo_curso"] = inep["codigo_curso"].astype("int64")
    inep["inep_ano_censo"] = inep["inep_ano_censo"].astype("int64")

    registros = []
    colunas_inep = [col for col in inep.columns if col != "codigo_curso"]

    for codigo_curso, grupo in inep.groupby("codigo_curso", sort=False):
        grupo = grupo.sort_values("inep_ano_censo", kind="mergesort")

        for ano_fies in anos_fies:
            exato = grupo[grupo["inep_ano_censo"] == ano_fies]

            if not exato.empty:
                linha = exato.iloc[-1]
                match_tipo = "ano_exato"

            else:
                passado = grupo[grupo["inep_ano_censo"] < ano_fies]

                if not passado.empty:
                    linha = passado.iloc[-1]
                    match_tipo = "fallback_passado"

                else:
                    futuro = grupo[grupo["inep_ano_censo"] > ano_fies]

                    if not futuro.empty:
                        linha = futuro.iloc[0]
                        match_tipo = "fallback_futuro"
                    else:
                        continue

            registro = {
                "codigo_curso": codigo_curso,
                "ano_fies_ref": ano_fies,
                "cine_match_tipo": match_tipo,
            }

            for col in colunas_inep:
                registro[col] = linha[col]

            registros.append(registro)

    if not registros:
        return pd.DataFrame(columns=["codigo_curso", "ano_fies_ref", "cine_match_tipo"] + colunas_inep)

    referencia = pd.DataFrame(registros)

    referencia["codigo_curso"] = pd.to_numeric(referencia["codigo_curso"], errors="coerce").astype("Int64")
    referencia["ano_fies_ref"] = pd.to_numeric(referencia["ano_fies_ref"], errors="coerce").astype("Int64")
    referencia["inep_ano_censo"] = pd.to_numeric(referencia["inep_ano_censo"], errors="coerce").astype("Int64")

    referencia = referencia.drop_duplicates(
        subset=["codigo_curso", "ano_fies_ref"],
        keep="last",
    ).reset_index(drop=True)

    return referencia


def salvar_auxiliar_cursos_validos(df: pd.DataFrame, nome_base: str) -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    mask = df[COL_CURSO].notna() & df[COL_CINE_NOME].notna()

    if not mask.any():
        return

    auxiliar = (
        df.loc[mask, [COL_CURSO, COL_CINE_NOME, COL_CINE_COD]]
        .drop_duplicates()
        .sort_values(by=[COL_CURSO], kind="mergesort")
        .reset_index(drop=True)
    )

    caminho = TEMP_DIR / f"dataset_auxiliar_curso_area_{nome_base}.csv"
    auxiliar.to_csv(caminho, index=False, encoding="utf-8")

    log(f"[OK] Auxiliar de cursos válidos salvo em: {caminho}")


def diagnosticar_sem_cine(df: pd.DataFrame, nome_base: str, momento: str) -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if COL_CINE_NOME not in df.columns:
        df[COL_CINE_NOME] = pd.NA

    if "_curso_norm" not in df.columns:
        df["_curso_norm"] = df[COL_CURSO].map(normalizar_nome_sem_acento).astype("string")

    mask = df[COL_CINE_NOME].isna()

    if mask.any():
        diagnostico = (
            df.loc[mask, [COL_CURSO, "_curso_norm"]]
            .fillna({COL_CURSO: "<NA>", "_curso_norm": "<NA>"})
            .value_counts()
            .reset_index(name="quantidade")
            .rename(columns={COL_CURSO: "nome_curso", "_curso_norm": "curso_normalizado"})
            .sort_values(by="quantidade", ascending=False, kind="mergesort")
        )
    else:
        diagnostico = pd.DataFrame(columns=["nome_curso", "curso_normalizado", "quantidade"])

    caminho = TEMP_DIR / f"analise_nomes_sem_cine_{nome_base}_{momento}.csv"
    diagnostico.to_csv(caminho, index=False, encoding="utf-8")

    log(f"[OK] Diagnóstico sem CINE salvo em: {caminho}")


def aplicar_mapa_manual(df: pd.DataFrame, mapa: dict, nome_base: str) -> tuple[pd.DataFrame, dict]:
    mapa_nome_exato, mapa_cod_exato, mapa_nome_norm, mapa_cod_norm = preparar_mapa_manual(mapa)

    if COL_CINE_NOME not in df.columns:
        df[COL_CINE_NOME] = pd.NA

    if COL_CINE_COD not in df.columns:
        df[COL_CINE_COD] = pd.NA

    if "cine_match_tipo" not in df.columns:
        df["cine_match_tipo"] = pd.NA

    df["_curso_exato"] = df[COL_CURSO].map(normalizar_nome).astype("string")
    df["_curso_norm"] = df[COL_CURSO].map(normalizar_nome_sem_acento).astype("string")

    sem_nome_antes = int(df[COL_CINE_NOME].isna().sum())
    sem_cod_antes = int(df[COL_CINE_COD].isna().sum())

    mask_corrigir = df[COL_CINE_NOME].isna() | df[COL_CINE_COD].isna()

    if mask_corrigir.any():
        nome_exato = df.loc[mask_corrigir, "_curso_exato"].map(mapa_nome_exato)
        cod_exato = df.loc[mask_corrigir, "_curso_exato"].map(mapa_cod_exato)

        nome_norm = df.loc[mask_corrigir, "_curso_norm"].map(mapa_nome_norm)
        cod_norm = df.loc[mask_corrigir, "_curso_norm"].map(mapa_cod_norm)

        nome_final = nome_exato.fillna(nome_norm)
        cod_final = cod_exato.fillna(cod_norm)

        idx_nome = nome_final[nome_final.notna()].index
        idx_cod = cod_final[cod_final.notna()].index

        idx_nome_alvo = df.index[df[COL_CINE_NOME].isna()].intersection(idx_nome)
        idx_cod_alvo = df.index[df[COL_CINE_COD].isna()].intersection(idx_cod)

        df.loc[idx_nome_alvo, COL_CINE_NOME] = nome_final.loc[idx_nome_alvo].astype("string")
        df.loc[idx_cod_alvo, COL_CINE_COD] = cod_final.loc[idx_cod_alvo].astype("string")

        idx_match = idx_nome.union(idx_cod)
        df.loc[idx_match, "cine_match_tipo"] = df.loc[idx_match, "cine_match_tipo"].fillna("manual_transform_layer_2")

        if "nome_curso_inep" in df.columns:
            idx_inep = df.index[df["nome_curso_inep"].isna()].intersection(idx_match)
            df.loc[idx_inep, "nome_curso_inep"] = df.loc[idx_inep, COL_CURSO]

    df[COL_CINE_COD] = df[COL_CINE_COD].map(normalizar_codigo_cine).astype("string")
    df[COL_CINE_NOME] = df[COL_CINE_NOME].map(normalizar_nome).astype("string")

    sem_nome_depois = int(df[COL_CINE_NOME].isna().sum())
    sem_cod_depois = int(df[COL_CINE_COD].isna().sum())

    df = df.drop(columns=["_curso_exato", "_curso_norm"], errors="ignore")

    return df, {
        "base": nome_base,
        "linhas": len(df),
        "sem_codigo_curso": int(df["codigo_curso"].isna().sum()) if "codigo_curso" in df.columns else len(df),
        "sem_cine_area_geral": sem_nome_depois,
        "sem_nome_cine_antes_manual": sem_nome_antes,
        "sem_nome_cine_depois_manual": sem_nome_depois,
        "sem_codigo_cine_antes_manual": sem_cod_antes,
        "sem_codigo_cine_depois_manual": sem_cod_depois,
        "corrigidos_nome_cine_manual": sem_nome_antes - sem_nome_depois,
        "corrigidos_codigo_cine_manual": sem_cod_antes - sem_cod_depois,
        "cursos_no_mapa_manual": len(mapa),
        "cursos_utilizaveis_no_mapa": len(mapa_nome_norm),
    }


def auditar_resultado(df: pd.DataFrame, nome_base: str, resumo_manual: dict) -> dict:
    distribuicao = (
        df["cine_match_tipo"]
        .fillna("sem_match")
        .value_counts(dropna=False)
        .to_dict()
        if "cine_match_tipo" in df.columns
        else {"sem_match": len(df)}
    )

    resultado = dict(resumo_manual)
    resultado.update({
        "ano_exato": int(distribuicao.get("ano_exato", 0)),
        "fallback_passado": int(distribuicao.get("fallback_passado", 0)),
        "fallback_futuro": int(distribuicao.get("fallback_futuro", 0)),
        "manual_transform_layer_2": int(distribuicao.get("manual_transform_layer_2", 0)),
        "sem_match": int(distribuicao.get("sem_match", 0)),
    })

    log(f"[AUDITORIA] {nome_base}: linhas totais: {resultado['linhas']}")
    log(f"[AUDITORIA] {nome_base}: linhas sem codigo_curso: {resultado['sem_codigo_curso']}")
    log(f"[AUDITORIA] {nome_base}: linhas sem CINE geral: {resultado['sem_cine_area_geral']}")
    log(f"[AUDITORIA] {nome_base}: distribuição match CINE: {distribuicao}")

    return resultado


def enriquecer_com_cine(
    arquivo_entrada: Path,
    arquivo_saida: Path,
    ano_col: str,
    nome_base: str,
    inep: pd.DataFrame,
    mapa_manual: dict,
) -> dict:
    log("=" * 80)
    log(f"CRUZAMENTO CINE: {nome_base}")
    log("=" * 80)

    df = preparar_base_fies(arquivo_entrada, ano_col)

    log(f"[INÍCIO] {arquivo_entrada.name} | linhas: {len(df)} | colunas: {len(df.columns)}")

    anos_fies = (
        df[ano_col]
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    log(f"[INFO] {nome_base}: anos encontrados para cruzamento: {sorted(anos_fies)}")

    referencia = construir_referencia_cine(inep, anos_fies)

    log(
        f"[INFO] {nome_base}: referência CINE construída | "
        f"linhas: {len(referencia)} | colunas: {len(referencia.columns)}"
    )

    out = df.merge(
        referencia,
        how="left",
        left_on=["codigo_curso", ano_col],
        right_on=["codigo_curso", "ano_fies_ref"],
        validate="m:1",
    )

    if "ano_fies_ref" in out.columns:
        out = out.drop(columns=["ano_fies_ref"])

    salvar_auxiliar_cursos_validos(out, nome_base)
    diagnosticar_sem_cine(out, nome_base, "antes_manual")

    out, resumo_manual = aplicar_mapa_manual(out, mapa_manual, nome_base)

    diagnosticar_sem_cine(out, nome_base, "depois_manual")

    out.to_parquet(arquivo_saida, index=False)

    log(f"[OK] Salvo em: {arquivo_saida}")
    log(f"[OK] Colunas finais: {len(out.columns)}")

    return auditar_resultado(out, nome_base, resumo_manual)


def salvar_resumo(registros: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(registros)
    df.to_csv(RESUMO_CRUZAMENTO_CINE, index=False, encoding="utf-8")
    log(f"[OK] Resumo salvo em: {RESUMO_CRUZAMENTO_CINE}")


def run() -> None:
    log("=" * 80)
    log("TRANSFORM: CRUZAMENTO FIES × INEP/CINE")
    log("=" * 80)

    inep = preparar_inep()

    log(f"[OK] Mestre INEP carregado | linhas: {len(inep)} | colunas: {len(inep.columns)}")

    registros = [
        enriquecer_com_cine(
            arquivo_entrada=ARQUIVO_INSCRICOES_UNIFICADAS,
            arquivo_saida=ARQUIVO_INSCRICOES_COM_CINE,
            ano_col="ano_processo_seletivo",
            nome_base="inscricoes",
            inep=inep,
            mapa_manual=MAPA_CORRECAO_MANUAL_INSCRICOES,
        ),
        enriquecer_com_cine(
            arquivo_entrada=ARQUIVO_OFERTAS_UNIFICADAS,
            arquivo_saida=ARQUIVO_OFERTAS_COM_CINE,
            ano_col="ano",
            nome_base="ofertas",
            inep=inep,
            mapa_manual=MAPA_CORRECAO_MANUAL_OFERTAS,
        ),
    ]

    salvar_resumo(registros)

    log("Cruzamento FIES × INEP/CINE concluído.")
