import pandas as pd
import configparser
from datetime import datetime, timedelta


def generate_summary_report(device_id, df):
    """
    Generates a summary report for a given device ID.
    """
    device_df = df[df['device_id'] == device_id]

    if device_df.empty:
        return f"# Relatório de Resumo para o Dispositivo: {device_id}\n\nDispositivo não encontrado."

    report = f"# Relatório de Resumo para o Dispositivo: {device_id}\n\n"

    # --- Análise de Keep Alives ---
    report += "## Análise de Keep Alives\n\n"
    device_df = device_df.sort_values(by='@timestamp')
    if 'f_cnt' in device_df.columns:
        device_df['f_cnt_diff'] = device_df['f_cnt'].diff()
        lost_packets = device_df[device_df['f_cnt_diff'] > 1]['f_cnt_diff'].sum() - len(device_df[device_df['f_cnt_diff'] > 1])
        report += f"- Pacotes perdidos (baseado no f_cnt): {lost_packets}\n"
    else:
        report += "- Não foi possível analisar a perda de pacotes (f_cnt não encontrado).\n"

    # --- Análise de Conectividade ---
    report += "\n## Análise de Conectividade\n\n"
    latest_record = device_df.iloc[-1]
    report += f"- RSSI: {latest_record.get('rssi', 'N/A')}\n"
    report += f"- SNR: {latest_record.get('snr', 'N/A')}\n"
    report += f"- RSRP: {latest_record.get('rsrp', 'N/A')}\n"
    report += f"- RSRQ: {latest_record.get('rsrq', 'N/A')}\n"
    report += f"- ECL: {latest_record.get('ecl', 'N/A')}\n"
    report += f"- TX Power: {latest_record.get('txPower', 'N/A')}\n"
    report += f"- Tempo de Registro: {latest_record.get('registrationTime', 'N/A')}\n"

    # --- Análise de Bateria ---
    report += "\n## Análise de Bateria\n\n"
    if 'battery_voltage' in device_df.columns:
        latest_voltage = latest_record.get('battery_voltage', 'N/A')
        report += f"- Última voltagem da bateria: {latest_voltage}\n"
        if isinstance(latest_voltage, (int, float)) and latest_voltage < 2.5:
            report += "- **Alerta: Nível de bateria crítico!**\n"
    else:
        report += "- Não foi possível analisar a bateria (battery_voltage não encontrado).\n"

    # --- Análise de Status do Sensor ---
    report += "\n## Análise de Status do Sensor\n\n"
    report += f"- Versão do Firmware: {latest_record.get('fw_app_version', 'N/A')}\n"
    report += f"- Status do Sensor: {latest_record.get('statusSensor', 'N/A')}\n"
    if latest_record.get('statusSensor') == 'ALERT':
        report += "- **Alerta: Falha de hardware detectada!**\n"

    return report

import sys

def main():
    """
    Generates a summary report for a given device ID.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    processed_payload_path = config['Paths']['processed_payload']

    try:
        df = pd.read_csv(processed_payload_path, low_memory=False)
        df['device_id'] = df['device_id'].astype(str).str.strip()
        print("Primeiros 5 device_ids do DataFrame:")
        print(df['device_id'].head())
    except FileNotFoundError:
        print(f"Error: The file {processed_payload_path} was not found.")
        return

    if len(sys.argv) < 2:
        print("Usage: python automated_report.py <device_id>")
        return

    device_id = sys.argv[1]

    report = generate_summary_report(device_id, df)

    with open(f"relatorio_{device_id}.md", "w") as f:
        f.write(report)

    print(f"Relatório salvo em relatorio_{device_id}.md")

if __name__ == "__main__":
    main()
