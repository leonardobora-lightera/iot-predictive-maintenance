import streamlit as st
import pandas as pd
import plotly.express as px
import configparser

ERROR_DESCRIPTIONS = {
    1: "APPMANAGER_KP_FAIL: Perda de keep alive",
    2: "APPMANAGER_ALERT_FAIL: Perda de alerta",
    3: "APPMANAGER_RETRANSMISSION_FAIL",
    4: "APPMANAGER_ERROR_MESSAGE_FAIL",
    5: "KP_UNCONFIRMED: Reset por kp não confirmado",
    6: "",
    7: "",
    8: "",
    9: "",
    10: "",
    11: "",
    12: "",
    13: "",
    14: "",
    15: "",
    16: "",
    17: "",
    18: "",
    19: "",
    20: "",
    21: "",
    22: "",
    23: "",
    24: "",
    25: "",
    26: "",
    27: "",
    28: "REGISTRATION_DENIED: Registro negado (CEREG=3)",
}

def main():
    st.set_page_config(page_title="Análise de Erros", layout="wide")
    st.title("Dashboard de Análise de Erros")

    config = configparser.ConfigParser()
    config.read('config.ini')
    processed_payload_path = config['Paths']['processed_payload']

    try:
        df = pd.read_csv(processed_payload_path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Error: The file {processed_payload_path} was not found.")
        return

    error_cols = [col for col in df.columns if col.startswith('buffer.error_codes.')]

    if not error_cols:
        st.warning("No error code columns found in the data.")
        return

    all_errors = df[error_cols].stack().dropna()
    all_errors = all_errors[all_errors != 0]

    if all_errors.empty:
        st.info("No errors found in the data.")
        return

    error_counts = all_errors.value_counts().reset_index()
    error_counts.columns = ['Error Code', 'Count']
    error_counts['Error Code'] = error_counts['Error Code'].astype(int)
    error_counts['Description'] = error_counts['Error Code'].map(ERROR_DESCRIPTIONS).fillna("Descrição não disponível")

    st.subheader("Contagem de Códigos de Erro")
    fig = px.bar(error_counts, x='Error Code', y='Count', hover_data=['Description'], title="Ocorrências de Códigos de Erro")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabela de Códigos de Erro")
    st.dataframe(error_counts, use_container_width=True)


if __name__ == "__main__":
    main()
