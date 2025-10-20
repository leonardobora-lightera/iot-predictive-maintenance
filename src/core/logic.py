import pandas as pd
from datetime import datetime, timedelta

import configparser

def load_data(use_clean_data=False):
    """
    Carrega os dados do CSV de forma otimizada, utilizando apenas as colunas necessárias
    e tipos de dados eficientes para reduzir o consumo de memória.
    
    Args:
        use_clean_data: If True, loads payloads_processed_clean.csv instead of payloads_processed.csv
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    if use_clean_data:
        # Try to load cleaned data first
        processed_payload_path = config['Paths']['processed_payload'].replace('.csv', '_clean.csv')
        import os
        if not os.path.exists(processed_payload_path):
            print(f"⚠️ Clean data not found at {processed_payload_path}, falling back to original data")
            processed_payload_path = config['Paths']['processed_payload']
        else:
            print(f"📊 Using cleaned data: {processed_payload_path}")
    else:
        processed_payload_path = config['Paths']['processed_payload']

    # Lista de colunas que são efetivamente usadas pela aplicação.
    # Ignorar colunas não utilizadas economiza uma quantidade significativa de memória.
    used_cols = [
        '@timestamp', 'device_id', 'msg_type', 'cliente', 'project_name',
        'registration_time', 'status_signal_quality', 'fw_app_version', 'fw_app_rc',
        'status_rssi', 'status_rsrp', 'status_rsrq', 'status_snr', 'status_ecl',
        'last_status', # Adicionado para exibição no dashboard geral
        'battery_voltage', # Adicionado para análise de bateria
        'buffer.error_codes.0', 'buffer.error_codes.1', 'buffer.error_codes.2',
        'buffer.error_codes.3', 'buffer.error_codes.4', 'buffer.error_codes.5',
        'buffer.error_codes.6', 'buffer.error_codes.7', 'buffer.error_codes.8',
        'buffer.error_codes.9', 'buffer.error_codes.10', 'buffer.error_codes.11',
        'buffer.error_codes.12', 'buffer.error_codes.13', 'buffer.error_codes.14',
        'buffer.error_codes.15', 'buffer.error_codes.16', 'buffer.error_codes.17',
        'buffer.error_codes.18', 'buffer.error_codes.19', 'buffer.error_codes.20',
        'buffer.error_codes.21', 'buffer.error_codes.22', 'buffer.error_codes.23',
        'buffer.error_codes.24', 'buffer.error_codes.25', 'buffer.error_codes.26',
        'buffer.error_codes.27', 'buffer.error_codes.28', 'buffer.error_codes.29',
        'buffer.total_errors',
        # Colunas de qualidade de sinal detalhado adicionadas para o template
        'apn.name', 'iccid', 'imsi', 'rssi', 'snr', 'txPower', 'ecl', 'rsrp', 'rsrq'
    ]

    # Dtypes otimizados: 'category' é excelente para colunas com baixa cardinalidade (poucos valores únicos)
    col_types = {
        'cliente': 'category',
        'project_name': 'category',
        'status_signal_quality': 'category',
        'msg_type': 'category',
        'fw_app_version': 'category',
        'fw_app_rc': 'category',
        'device_id': str
    }

    try:
        # A função lambda em usecols evita erros se uma coluna não for encontrada no CSV
        df = pd.read_csv(
            processed_payload_path,
            usecols=lambda c: c in used_cols,
            dtype=col_types,
            low_memory=False
        )
        
        if '@timestamp' in df.columns:
            df['@timestamp'] = pd.to_datetime(df['@timestamp'])
        
        return df

    except Exception as e:
        print(f"AVISO: Não foi possível aplicar o filtro de colunas otimizado. Carregando o CSV completo. Erro: {e}")
        # Fallback para o método original se a otimização falhar
        df = pd.read_csv(processed_payload_path, low_memory=False)
        if '@timestamp' in df.columns:
            df['@timestamp'] = pd.to_datetime(df['@timestamp'])
        return df

# O restante do arquivo permanece o mesmo...

def analyze_device_example(df, device_id, columns=None, ascending=True):
    """
    Analisa um device específico após aplicar as métricas
    
    Parâmetros:
    - df: DataFrame com os dados
    - device_id: ID do dispositivo a analisar
    - columns: Lista de colunas para exibir (além do @timestamp). Se None, exibe todas
    - ascending: True para ordem crescente, False para decrescente do @timestamp
    """
    if 'lost_payloads_percent' not in df.columns:
        return None
    
    df['device_id'] = df['device_id'].astype(str).str.strip()
    device_id_str = str(device_id).strip()
    
    device_data = df[df['device_id'] == device_id_str].copy()
    
    if len(device_data) == 0:
        device_data = df[df['device_id'].str.lstrip('0') == device_id_str.lstrip('0')].copy()
    
    if len(device_data) == 0:
        return None
    
    total_records = len(device_data)
    final_percent = device_data['lost_payloads_percent'].iloc[-1] if 'lost_payloads_percent' in device_data.columns else 0
    alert_count = (device_data['lost_payloads_status'] == 'ALERT').sum() if 'lost_payloads_status' in device_data.columns else 0
    
    if '@timestamp' in device_data.columns:
        device_data = device_data.sort_values(by='@timestamp', ascending=ascending)
    
    if columns:
        display_columns = ['@timestamp']
        for col in columns:
            if col in device_data.columns and col != '@timestamp':
                display_columns.append(col)
        
        missing_columns = [col for col in columns if col not in device_data.columns]
        
        sample_data = device_data[display_columns]
    else:
        sample_data = device_data
    
    return device_data

def calculate_days_without_communication(df):
    """Calcula dias sem comunicação baseado no timestamp mais recente"""
    df_result = df.copy()
    df_result['@timestamp'] = pd.to_datetime(df_result['@timestamp'])
    
    latest_communication = df_result.groupby('device_id')['@timestamp'].max().reset_index()
    latest_communication['days_without_comm'] = (datetime.now() - latest_communication['@timestamp']).dt.days
    
    df_result = df_result.merge(latest_communication[['device_id', 'days_without_comm']], on='device_id', how='left')
    
    return df_result

def total_msg_type6_week(df):
    """
    Retorna o total de mensagens com msg_type == 6 na última semana (todos os devices).
    """
    df_copy = df.copy()
    df_copy['@timestamp'] = pd.to_datetime(df_copy['@timestamp'])
    df_copy['msg_type'] = pd.to_numeric(df_copy['msg_type'], errors='coerce').astype('Int64')
    last_week = datetime.now() - timedelta(days=7)
    total = df_copy[
        (df_copy['@timestamp'] >= last_week) & (df_copy['msg_type'] == 6)
    ].shape[0]
    return total

def filter_msg_type6_week(df, days=7):
    df_copy = df.copy()
    df_copy['@timestamp'] = pd.to_datetime(df_copy['@timestamp'])
    df_copy['msg_type'] = pd.to_numeric(df_copy['msg_type'], errors='coerce').astype('Int64')
    last_period = datetime.now() - timedelta(days=days)
    return df_copy[(df_copy['@timestamp'] >= last_period) & (df_copy['msg_type'] == 6)]

def count_msg_type6_week(df, days=7):
    """
    Adiciona coluna indicando, para cada device_id, quantas mensagens msg_type == 6
    ele teve nos últimos dias especificados (usando filter_msg_type6_week).
    """
    df_result = df.copy()
    df_filtered = filter_msg_type6_week(df_result, days=days)
    msg_type6_counts = df_filtered.groupby('device_id').size()
    df_result['msg_type6_week'] = df_result['device_id'].map(msg_type6_counts).fillna(0).astype(int)
    return df_result

def mean_registration_time_week(df, days=7):
    """
    Calcula a média do tempo de registro nos últimos dias especificados para cada device_id.
    Retorna o dataframe original acrescido da coluna 'registration_time_mean_week'.
    """
    df_result = df.copy()
    df_result['@timestamp'] = pd.to_datetime(df_result['@timestamp'])

    last_period = datetime.now() - timedelta(days=days)
    period_df = df_result[df_result['@timestamp'] >= last_period].copy()

    if 'registration_time' not in period_df.columns:
        df_result['registration_time_mean_week'] = None
        return df_result

    mean_reg_time = (
        period_df.groupby('device_id')['registration_time']
        .mean()
        .reset_index(name='registration_time_mean_week')
    )

    df_result = df_result.merge(mean_reg_time, on='device_id', how='left')

    return df_result

def payloads_por_dia_semana(df, days=7):
    """
    Para cada device_id e dia dos últimos dias especificados, calcula:
    - 'payloads_count_dia': quantos registros msg_type=1 por device/dia
    - 'payloads_dia_status': 'ok' se >=4, 'alert' se <4 (por dia)
    - 'payloads_perdidos_dia': quanto faltou para 4 nesse dia (mínimo 0)
    Adiciona essas colunas no df original, de acordo com device_id e data.
    """
    df_result = df.copy()
    df_result['@timestamp'] = pd.to_datetime(df_result['@timestamp'])
    df_result['date'] = df_result['@timestamp'].dt.date

    last_period = datetime.now().date() - timedelta(days=days)
    period_df = df_result[df_result['date'] >= last_period].copy()

    period_df['msg_type'] = pd.to_numeric(period_df['msg_type'], errors='coerce')
    msg1 = period_df[period_df['msg_type'] == 1]

    counts = (
        msg1
        .groupby(['device_id', 'date'])
        .size()
        .reset_index(name='payloads_count_dia')
    )
    counts['payloads_dia_status'] = counts['payloads_count_dia'].apply(lambda x: 'ok' if x >= 4 else 'alert')
    counts['payloads_perdidos_dia'] = counts['payloads_count_dia'].apply(lambda x: max(0, 4-x))

    df_result = df_result.merge(
        counts,
        on=['device_id', 'date'],
        how='left'
    )

    df_result['payloads_count_dia'] = df_result['payloads_count_dia'].fillna(0).astype(int)
    df_result['payloads_dia_status'] = df_result['payloads_dia_status'].fillna('alert')
    df_result['payloads_perdidos_dia'] = df_result['payloads_perdidos_dia'].fillna(4).astype(int)

    return df_result

def generate_sensor_report(df, device_ids):
    """
    Gera um relatório para uma lista de devices.
    """
    report_data = []
    
    df['device_id'] = df['device_id'].astype(str).str.strip()

    for device_id in device_ids:
        device_id_str = str(device_id).strip()
        
        device_data = df[df['device_id'] == device_id_str].copy()
        
        if len(device_data) == 0:
            device_data = df[df['device_id'].str.lstrip('0') == device_id_str.lstrip('0')].copy()

        if len(device_data) == 0:
            report_data.append({
                'Device_ID': device_id_str,
                'Status': 'Não encontrado',
                'Total de Registros': 0,
                'Msg_type=6 (última semana)': 0,
                'Última comunicação': 'N/A',
                'Dias sem comunicação': 'N/A',
                'Versão Firmware': 'N/A',
                'Cliente': 'N/A',
                'Projeto': 'N/A'
            })
            continue

        total_records = len(device_data)
        msg_type6_count = device_data['msg_type6_week'].iloc[0] if 'msg_type6_week' in device_data.columns and not device_data.empty else 0
        
        latest_record = device_data.sort_values(by='@timestamp', ascending=False).iloc[0]
        
        days_without_comm = (datetime.now() - latest_record['@timestamp']).days
        
        report_row = {
            'Device_ID': device_id_str,
            'Status': 'Encontrado',
            'Total de Registros': total_records,
            'Msg_type=6 (última semana)': msg_type6_count,
            'Última comunicação': latest_record['@timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Dias sem comunicação': days_without_comm,
            'Versão Firmware': latest_record.get('fw_app_version', 'N/A'),
            'Cliente': latest_record.get('cliente', 'N/A'),
            'Projeto': latest_record.get('project_name', 'N/A')
        }
        report_data.append(report_row)
        
    return pd.DataFrame(report_data)

def analyze_last_week(df, days=7):
    """
    Analisa os dados dos últimos dias especificados para o dashboard geral.
    """
    df_copy = df.copy()
    df_copy['@timestamp'] = pd.to_datetime(df_copy['@timestamp'])
    last_period = datetime.now() - timedelta(days=days)
    
    period_df = df_copy[df_copy['@timestamp'] >= last_period].copy()
    period_df['date'] = period_df['@timestamp'].dt.date

    if 'payloads_perdidos_dia' not in period_df.columns:
        period_df = payloads_por_dia_semana(period_df, days=days)
    
    if 'registration_time' not in period_df.columns:
        if 'registration_time' not in period_df.columns:
            period_df['registration_time'] = 0

    lost_payloads_sensors_daily = period_df[period_df['payloads_perdidos_dia'] > 0].groupby('date')['device_id'].nunique().reset_index(name='lost_payload_sensors')
    slow_reg_sensors_daily = period_df[period_df['registration_time'] > 120].groupby('date')['device_id'].nunique().reset_index(name='slow_reg_sensors')

    period_df['msg_type'] = pd.to_numeric(period_df['msg_type'], errors='coerce').astype('Int64')
    msg_type6_sensors_daily = period_df[period_df['msg_type'] == 6].groupby('date')['device_id'].nunique().reset_index(name='msg_type6_sensors')

    analysis_df = pd.merge(lost_payloads_sensors_daily, slow_reg_sensors_daily, on='date', how='outer')
    analysis_df = pd.merge(analysis_df, msg_type6_sensors_daily, on='date', how='outer')
    analysis_df = analysis_df.fillna(0)

    return analysis_df

def get_sensors_for_analysis(df, sensores_chamado, days=7):
    """
    Filtra sensores para a tabela de análise.
    """
    df_analysis = df.copy()
    
    sensores_chamado_set = set(sensores_chamado)
    df_analysis = df_analysis[~df_analysis['device_id'].isin(sensores_chamado_set)]
    
    df_analysis['@timestamp'] = pd.to_datetime(df_analysis['@timestamp'])
    last_period = datetime.now() - timedelta(days=days)
    df_period = df_analysis[df_analysis['@timestamp'] >= last_period]
    latest_records = df_period.sort_values('@timestamp').groupby('device_id').tail(1)

    cond1 = latest_records['msg_type6_week'] > 0
    cond2 = latest_records['registration_time_mean_week'] > 120
    cond3 = latest_records['payloads_perdidos_dia'] > 0
    cond4 = latest_records['status_signal_quality'].isin(['fraco', 'Sinal muito fraco', 'Sinal fraco'])
    
    analysis_sensors = latest_records[cond1 | cond2 | cond3 | cond4]
    
    return analysis_sensors

def get_communication_status(df, device_ids):
    """
    Determina o status de comunicação para uma lista de devices.
    """
    status_data = []
    df['device_id'] = df['device_id'].astype(str).str.strip()
    now = datetime.now()

    for device_id in device_ids:
        device_id_str = str(device_id).strip()
        device_data = df[df['device_id'] == device_id_str]

        if device_data.empty:
            status_data.append({
                'Device_ID': device_id_str,
                'Status de Comunicação': 'Sem registros',
                'Última Comunicação': 'N/A'
            })
            continue

        latest_record = device_data.sort_values(by='@timestamp', ascending=False).iloc[0]
        last_comm_time = latest_record['@timestamp']
        time_delta = now - last_comm_time

        if time_delta < timedelta(hours=2):
            status = "Comunicando normalmente"
        elif time_delta < timedelta(hours=4):
            status = "Atenção: Última comunicação entre 2 e 4 horas"
        else:
            status = "Crítico: Última comunicação há mais de 4 horas"

        status_data.append({
            'Device_ID': device_id_str,
            'Status de Comunicação': status,
            'Última Comunicação': last_comm_time.strftime('%Y-%m-%d %H:%M:%S')
        })

    return pd.DataFrame(status_data)

def generate_ticket_report(df, device_ids):
    """
    Gera um relatório detalhado para uma lista de devices de um chamado.
    """
    report_data = []
    df['device_id'] = df['device_id'].astype(str).str.strip()

    if 'payloads_perdidos_dia' not in df.columns:
        df = payloads_por_dia_semana(df, days=7)
    if 'registration_time_mean_week' not in df.columns:
        df = mean_registration_time_week(df, days=7)
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['@timestamp']).dt.date


    for device_id in device_ids:
        device_id_str = str(device_id).strip()
        device_data = df[df['device_id'] == device_id_str].copy()

        if device_data.empty:
            report_data.append({
                'Device_ID': device_id_str,
                'Cliente': 'N/A',
                'Projeto': 'N/A',
                'Versão Firmware': 'N/A',
                'fw_app_rc': 'N/A',
                'Última Com. (msg_type=1)': 'N/A',
                'Última Com. (msg_type=6)': 'N/A',
                'Status Payloads (7 dias)': 'Sem registros',
                'Tempo Médio de Registro (Semana)': 'N/A',
                'Dias sem comunicação': 'N/A'
            })
            continue

        latest_record = device_data.sort_values(by='@timestamp', ascending=False).iloc[0]
        cliente = latest_record.get('cliente', 'N/A')
        projeto = latest_record.get('project_name', 'N/A')
        fw_version = latest_record.get('fw_app_version', 'N/A')
        fw_app_rc = latest_record.get('fw_app_rc', 'N/A')

        device_data['msg_type'] = pd.to_numeric(device_data['msg_type'], errors='coerce').astype('Int64')
        
        last_comm_msg1 = device_data[device_data['msg_type'] == 1]['@timestamp'].max()
        last_comm_msg6 = device_data[device_data['msg_type'] == 6]['@timestamp'].max()
        
        days_without_comm = (datetime.now() - pd.to_datetime(latest_record['@timestamp'])).days

        # --- Lógica Refatorada para Status de Payload ---
        seven_days_ago = datetime.now().date() - timedelta(days=7)
        today = datetime.now().date()
        
        # Cria um range completo de datas para o período
        all_days = pd.date_range(start=seven_days_ago, end=today, freq='D').date
        
        # Filtra os dados do dispositivo para o período e obtém contagens únicas por dia
        device_data_period = device_data[device_data['date'].isin(all_days)]
        daily_counts = device_data_period[['date', 'payloads_count_dia']].drop_duplicates('date').set_index('date')
        
        # Verifica se há algum dado no período
        if device_data_period.empty:
            payload_status = "Sem dados nos últimos 7 dias"
        else:
            issues = []
            all_days_ok = True
            
            # Itera sobre cada dia na janela de tempo
            for day in sorted(all_days, reverse=True):
                if day in daily_counts.index:
                    count = daily_counts.loc[day, 'payloads_count_dia']
                    if count < 4:
                        all_days_ok = False
                        date_str = pd.to_datetime(day).strftime('%d/%m')
                        issues.append(f'{int(count)}/4 em {date_str}')
                else:
                    # Se o dia não está nos dados, significa que 0 payloads foram recebidos
                    all_days_ok = False
                    date_str = pd.to_datetime(day).strftime('%d/%m')
                    issues.append(f'0/4 em {date_str}')

            if all_days_ok:
                payload_status = "Normal (4/4 nos últimos 7 dias)"
            else:
                payload_status = "Falha: " + ", ".join(issues)

        avg_reg_time = device_data.sort_values(by='@timestamp', ascending=False).iloc[0].get('registration_time_mean_week')
        if pd.notna(avg_reg_time):
             avg_reg_time_str = f"{avg_reg_time:.2f}s"
        else:
             avg_reg_time_str = "N/A"

        report_data.append({
            'Device_ID': device_id_str,
            'Cliente': cliente,
            'Projeto': projeto,
            'Versão Firmware': fw_version,
            'fw_app_rc': fw_app_rc,
            'Última Com. (msg_type=1)': pd.to_datetime(last_comm_msg1).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(last_comm_msg1) else 'Nenhuma',
            'Última Com. (msg_type=6)': pd.to_datetime(last_comm_msg6).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(last_comm_msg6) else 'Nenhuma',
            'Status Payloads (7 dias)': payload_status,
            'Tempo Médio de Registro (Semana)': avg_reg_time_str,
            'Dias sem comunicação': days_without_comm
        })

    return pd.DataFrame(report_data)
