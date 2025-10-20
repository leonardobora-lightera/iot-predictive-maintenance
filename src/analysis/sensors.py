import streamlit as st
import pandas as pd
from core import logic as dp
import re
from datetime import datetime, timedelta

def get_battery_status(voltage):
    """Classifica a tensão da bateria em um status legível."""
    if pd.isna(voltage) or voltage == 0:
        return 'N/A'
    # Converte mV para V se o valor for muito alto (ex: 3500mV)
    if voltage > 5:
        voltage = voltage / 1000
    
    if voltage >= 3.0:
        return f'{voltage:.2f}V (OK)'
    elif 2.5 <= voltage < 3.0:
        return f'{voltage:.2f}V (Alerta)'
    else:
        return f'{voltage:.2f}V (Crítico)'

def generate_detailed_text_report(df, device_ids):
    """Gera um relatório de texto detalhado para uma lista de IDs de dispositivos."""
    report_lines = []
    
    # Garante que a coluna de device_id no DF principal seja string para comparação
    df['device_id'] = df['device_id'].astype(str).str.strip()

    for device_id in device_ids:
        device_id_str = str(device_id).strip()
        report_lines.append(f"--- Relatório para o Device ID: {device_id_str} ---")
        
        device_df = df[df['device_id'] == device_id_str]

        if device_df.empty:
            report_lines.append("Device não encontrado na base de dados.\n")
            continue

        device_df = device_df.sort_values(by='@timestamp', ascending=False)
        latest_record = device_df.iloc[0]

        # 1. Última Comunicação
        last_comm = pd.to_datetime(latest_record['@timestamp'])
        report_lines.append(f"Última Comunicação: {last_comm.strftime('%d/%m/%Y %H:%M')}")

        # 2. Contagem por Tipo de Mensagem (msg_type)
        report_lines.append("Contagem por Tipo de Mensagem (msg_type):")
        if 'msg_type' in device_df.columns:
            msg_type_counts = device_df['msg_type'].value_counts()
            for msg_type, count in msg_type_counts.items():
                report_lines.append(f"- msgtype = {int(msg_type)} : {count} registros")
        else:
            report_lines.append("- Coluna 'msg_type' não encontrada.")
        report_lines.append("---")

        # 3. Ocorrências de Error Codes
        min_date = device_df['@timestamp'].min().strftime('%d/%m/%Y')
        max_date = device_df['@timestamp'].max().strftime('%d/%m/%Y')
        report_lines.append(f"Ocorrências de Error Codes (de {min_date} a {max_date}):")
        error_cols = [col for col in device_df.columns if col.startswith('buffer.error_codes.')]
        if error_cols:
            all_errors = device_df[error_cols].stack().dropna()
            all_errors = all_errors[all_errors != 0]
            if not all_errors.empty:
                error_counts = all_errors.value_counts()
                for code, count in error_counts.items():
                    report_lines.append(f"- O erro {int(code)} ocorreu {count} vezes")
            else:
                report_lines.append("- Nenhum erro registrado para este dispositivo.")
        else:
            report_lines.append("- Colunas de error code (buffer.error_codes.*) não encontradas.")
        report_lines.append("---")

        # 4. Últimos registros de qualidade
        def get_last_value(col_name, default='N/A'):
            if col_name in latest_record and pd.notna(latest_record[col_name]):
                return latest_record[col_name]
            return default

        report_lines.append(f"Último registro Qualidade do Sinal: {get_last_value('status_signal_quality')}")
        report_lines.append(f"Último RSSI: {get_last_value('status_rssi')}")
        report_lines.append(f"Último ECL: {get_last_value('status_ecl')}")
        report_lines.append(f"Último Registration TIME: {get_last_value('registration_time')}")

        # 5. Tempo Médio de Registro
        if 'registration_time' in device_df.columns:
            mean_reg_time = pd.to_numeric(device_df['registration_time'], errors='coerce').dropna().mean()
            report_lines.append(f"Tempo Médio de Registro: {mean_reg_time:.2f} segundos")
        else:
            report_lines.append("Tempo Médio de Registro: N/A")
        report_lines.append("---")

        # 6. Análise de Bateria
        report_lines.append("Análise de Bateria:")
        if 'battery_voltage' in device_df.columns and not device_df['battery_voltage'].dropna().empty:
            last_voltage = get_last_value('battery_voltage', 0)
            report_lines.append(f"- Última Tensão Registrada: {get_battery_status(last_voltage)}")
            
            # Análise de variação nos últimos 30 dias
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_battery_data = device_df[device_df['@timestamp'] >= thirty_days_ago]['battery_voltage'].dropna()
            if len(recent_battery_data) > 1:
                initial_voltage = recent_battery_data.iloc[-1] # Mais antigo
                final_voltage = recent_battery_data.iloc[0]   # Mais recente
                variation = final_voltage - initial_voltage
                report_lines.append(f"- Variação nos últimos 30 dias: {variation:.3f}V (de {initial_voltage:.2f}V para {final_voltage:.2f}V)")
            else:
                report_lines.append("- Não há dados suficientes nos últimos 30 dias para calcular a variação.")
        else:
            report_lines.append("- Nenhuma informação de bateria encontrada.")

        report_lines.append("\n") # Espaço entre relatórios

    return "\n".join(report_lines)

def show_sensor_report(df):
    st.header("📋 Relatório de Sensores")
    
    device_ids_input = st.text_area(
        "Insira os Device IDs (separados por vírgula, espaço ou nova linha):", 
        height=150, 
        placeholder="861275072310192, 861275072465673, ...",
        key='device_ids_report_input'
    )
    
    # Processar IDs para ter a lista pronta
    device_ids = [d.strip() for d in re.split(r'[\s,]+', device_ids_input) if d.strip()]

    col1, col2 = st.columns(2)

    # Botão para gerar relatório em tabela (CSV)
    with col1:
        if st.button("Gerar Relatório em Tabela (CSV)", key='generate_csv_report_button', use_container_width=True):
            if device_ids:
                with st.spinner("Gerando relatório em tabela..."):
                    report_df = dp.generate_sensor_report(df, device_ids)
                    st.session_state.report_df = report_df # Salva no session_state para download
            else:
                st.warning("Por favor, insira pelo menos um Device ID.")

    # Botão para gerar relatório em texto (TXT)
    with col2:
        if st.button("Gerar Relatório Detalhado (TXT)", key='generate_txt_report_button', use_container_width=True):
            if device_ids:
                with st.spinner("Gerando relatório de texto detalhado..."):
                    report_text = generate_detailed_text_report(df, device_ids)
                    st.session_state.report_text = report_text # Salva no session_state
            else:
                st.warning("Por favor, insira pelo menos um Device ID.")

    # Exibição do relatório em tabela (se gerado)
    if 'report_df' in st.session_state and not st.session_state.report_df.empty:
        st.markdown("### Relatório em Tabela")
        st.dataframe(st.session_state.report_df, use_container_width=True)
        csv_report = st.session_state.report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Tabela (CSV)", 
            data=csv_report, 
            file_name="sensor_report.csv", 
            mime="text/csv"
        )
        st.markdown("---")

    # Exibição do relatório em texto (se gerado)
    if 'report_text' in st.session_state and st.session_state.report_text:
        st.markdown("### Relatório Detalhado em Texto")
        st.code(st.session_state.report_text, language='text')
        st.download_button(
            label="📥 Download Relatório (TXT)",
            data=st.session_state.report_text.encode('utf-8'),
            file_name="relatorio_detalhado_sensores.txt",
            mime="text/plain",
        )
        st.markdown("---")

    if not device_ids:
        st.info("👈 Insira os Device IDs na área de texto e clique em um dos botões para gerar um relatório.")
        st.subheader("📋 Preview dos Dados")
        st.dataframe(df.head(100), use_container_width=True)
