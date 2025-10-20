import streamlit as st
import pandas as pd
import configparser

THRESHOLDS = {
    'luminosity': 2,
    'accelerometer': 20,
    'powermeter': -28,
    'temperature': 70,
    'battery': 2.50
}

def main():
    st.set_page_config(page_title="Análise de Thresholds", layout="wide")
    st.title("Dashboard de Análise de Thresholds")

    config = configparser.ConfigParser()
    config.read('config.ini')
    processed_payload_path = config['Paths']['processed_payload']

    try:
        df = pd.read_csv(processed_payload_path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Error: The file {processed_payload_path} was not found.")
        return

    st.sidebar.header("Filtros")
    device_id = st.sidebar.text_input("Filtrar por Device ID (deixe em branco para todos)")

    if device_id:
        df = df[df['device_id'] == device_id]

    st.subheader("Violações de Threshold")

    violations = []

    if 'Lux' in df.columns:
        lux_violations = df[df['Lux'] > THRESHOLDS['luminosity']]
        for index, row in lux_violations.iterrows():
            violations.append([row['device_id'], 'Luminosidade', row['Lux'], THRESHOLDS['luminosity'], row['@timestamp']])

    if 'Pitch_Acelerometro' in df.columns:
        pitch_violations = df[df['Pitch_Acelerometro'].abs() > THRESHOLDS['accelerometer']]
        for index, row in pitch_violations.iterrows():
            violations.append([row['device_id'], 'Acelerômetro (Pitch)', row['Pitch_Acelerometro'], THRESHOLDS['accelerometer'], row['@timestamp']])

    if 'Roll_Acelerometro' in df.columns:
        roll_violations = df[df['Roll_Acelerometro'].abs() > THRESHOLDS['accelerometer']]
        for index, row in roll_violations.iterrows():
            violations.append([row['device_id'], 'Acelerômetro (Roll)', row['Roll_Acelerometro'], THRESHOLDS['accelerometer'], row['@timestamp']])

    if 'Potencia_Optica_1490nm' in df.columns:
        power_violations = df[df['Potencia_Optica_1490nm'] < THRESHOLDS['powermeter']]
        for index, row in power_violations.iterrows():
            violations.append([row['device_id'], 'Powermeter', row['Potencia_Optica_1490nm'], THRESHOLDS['powermeter'], row['@timestamp']])

    if 'Temperatura_C' in df.columns:
        temp_violations = df[df['Temperatura_C'] > THRESHOLDS['temperature']]
        for index, row in temp_violations.iterrows():
            violations.append([row['device_id'], 'Temperatura', row['Temperatura_C'], THRESHOLDS['temperature'], row['@timestamp']])

    if 'Bateria' in df.columns:
        battery_violations = df[df['Bateria'] < THRESHOLDS['battery']]
        for index, row in battery_violations.iterrows():
            violations.append([row['device_id'], 'Bateria', row['Bateria'], THRESHOLDS['battery'], row['@timestamp']])

    if violations:
        violations_df = pd.DataFrame(violations, columns=['Device ID', 'Métrica', 'Valor', 'Threshold', 'Timestamp'])
        st.dataframe(violations_df, use_container_width=True)
    else:
        st.info("Nenhuma violação de threshold encontrada para os filtros selecionados.")

if __name__ == "__main__":
    main()
