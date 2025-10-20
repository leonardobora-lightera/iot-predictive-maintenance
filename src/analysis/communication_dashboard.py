import streamlit as st
import pandas as pd
import configparser
from datetime import datetime, timedelta

def main():
    st.set_page_config(page_title="Detector de Falha de Comunicação", layout="wide")
    st.title("Dashboard de Falha de Comunicação")

    config = configparser.ConfigParser()
    config.read('config.ini')
    processed_payload_path = config['Paths']['processed_payload']

    try:
        df = pd.read_csv(processed_payload_path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Error: The file {processed_payload_path} was not found.")
        return

    df['@timestamp'] = pd.to_datetime(df['@timestamp'])

    latest_communication = df.groupby('device_id')['@timestamp'].max()

    twenty_four_hours_ago = datetime.now() - timedelta(hours=24)

    silent_sensors = latest_communication[latest_communication < twenty_four_hours_ago].reset_index()
    silent_sensors.columns = ['Device ID', 'Última Comunicação']

    st.subheader("Sensores Silenciosos (sem payload nas últimas 24 horas)")

    st.sidebar.header("Filtros")
    device_id = st.sidebar.text_input("Filtrar por Device ID (deixe em branco para todos)")

    if device_id:
        silent_sensors = silent_sensors[silent_sensors['Device ID'] == device_id]

    if silent_sensors.empty:
        st.info("Nenhum sensor silencioso encontrado para os filtros selecionados.")
    else:
        st.dataframe(silent_sensors, use_container_width=True)

if __name__ == "__main__":
    main()
