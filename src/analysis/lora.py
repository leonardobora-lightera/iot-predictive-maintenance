import streamlit as st
import pandas as pd
import plotly.express as px
import os

import configparser

# Define o caminho para o arquivo de dados
config = configparser.ConfigParser()
config.read('config.ini')
LORA_DATA_PATH = config['Paths']['lora_payload']

@st.cache_data
def load_lora_data():
    """Carrega os dados do arquivo CSV de dados LoRa."""
    try:
        df = pd.read_csv(LORA_DATA_PATH)
        # Converte colunas de data/hora e numéricas que podem ser lidas como object
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
        numeric_cols = ['RSSI', 'SNR', 'EyON_RSSI', 'Potencia_Optica_1490nm', 'Pitch_Acelerometro', 'Roll_Acelerometro', 'Bateria', 'Lux', 'Temperatura_C']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados LoRa não encontrado em: {LORA_DATA_PATH}")
        st.info("Por favor, execute o script 'Query AWS (LoRa)' na página '⚙️ Atualizar Base de Dados' para gerar o arquivo.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os dados LoRa: {e}")
        return None

def show_lora_analysis_page():
    """Exibe a página de análise de dados LoRa."""
    st.header("🛰️ Análise de Dados LoRaWAN")
    
    df_lora = load_lora_data()

    if df_lora is None or df_lora.empty:
        st.warning("Nenhum dado LoRa para analisar.")
        return

    st.sidebar.subheader("Filtros de Análise LoRa")
    
    # Filtro de data
    min_date = df_lora['@timestamp'].min().date()
    max_date = df_lora['@timestamp'].max().date()

    # Define o padrão para os últimos 7 dias
    default_start_date = max_date - pd.Timedelta(days=6)

    # Garante que a data de início padrão não seja anterior à data mínima disponível
    if default_start_date < min_date:
        default_start_date = min_date
    
    start_date, end_date = st.sidebar.date_input(
        "Selecione o Período",
        (default_start_date, max_date), # Define o valor padrão
        min_value=min_date,
        max_value=max_date,
        key="lora_date_filter"
    )

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    df_filtered = df_lora[(df_lora['@timestamp'] >= start_datetime) & (df_lora['@timestamp'] < end_datetime)]

    if df_filtered.empty:
        st.warning("Nenhum dado encontrado para o período selecionado.")
        return

    # Métricas principais
    st.subheader("Métricas Principais no Período")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Uplinks", f"{len(df_filtered):,}")
    col2.metric("Gateways Únicos", f"{df_filtered['Gateway_ID'].nunique()}")
    col3.metric("Contador de Uplinks (max)", f"{df_filtered['Contador'].max() if 'Contador' in df_filtered.columns else 'N/A'}")

    st.markdown("---")

    # Tabela de dados brutos
    st.subheader("Tabela de Dados")
    st.info("A tabela abaixo exibe os dados brutos para o período selecionado.")
    st.dataframe(df_filtered.sort_values('@timestamp', ascending=False), use_container_width=True)
