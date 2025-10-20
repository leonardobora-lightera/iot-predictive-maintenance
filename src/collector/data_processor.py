import configparser

# Opt-in para o comportamento futuro do pandas, eliminando o FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# --- Funções de Processamento Otimizadas ---

def apply_signal_status_vectorized(df):
    """Aplica classificação de status de sinal de forma vetorizada com np.select."""
    df_out = df.copy()
    if 'rssi' in df_out.columns:
        conditions = [df_out['rssi'] > -70, df_out['rssi'] > -85, df_out['rssi'] > -100]
        choices = ['excelente', 'bom', 'razoável']
        df_out['status_rssi'] = np.select(conditions, choices, default='fraco')
        df_out.loc[df_out['rssi'].isna(), 'status_rssi'] = None
    if 'rsrp' in df_out.columns:
        conditions = [df_out['rsrp'] >= -100, df_out['rsrp'] >= -110]
        choices = ['forte', 'médio']
        df_out['status_rsrp'] = np.select(conditions, choices, default='fraco')
        df_out.loc[df_out['rsrp'].isna(), 'status_rsrp'] = None
    if 'rsrq' in df_out.columns:
        conditions = [df_out['rsrq'] >= -7, df_out['rsrq'] > -11]
        choices = ['forte', 'médio']
        df_out['status_rsrq'] = np.select(conditions, choices, default='fraco')
        df_out.loc[df_out['rsrq'].isna(), 'status_rsrq'] = None
    if 'snr' in df_out.columns:
        conditions = [df_out['snr'] >= 3, df_out['snr'] > -3]
        choices = ['forte', 'médio']
        df_out['status_snr'] = np.select(conditions, choices, default='fraco')
        df_out.loc[df_out['snr'].isna(), 'status_snr'] = None
    if 'ecl' in df_out.columns:
        df_out['status_ecl'] = df_out['ecl'].map({0: 'baixo', 1: 'médio', 2: 'alto'})

    status_cols = [col for col in ['status_rsrp', 'status_snr', 'status_rsrq'] if col in df_out.columns]
    
    if status_cols:
        df_s = df_out[status_cols]
        num_valid = df_s.notna().sum(axis=1)
        num_fraco = (df_s == 'fraco').sum(axis=1)
        num_medio = (df_s == 'médio').sum(axis=1)
        num_forte = (df_s == 'forte').sum(axis=1)

        # A ordem das condições simula a lógica if/elif do código original
        conditions = [
            (num_valid > 0) & (num_fraco == num_valid),   # all 'fraco'
            (num_valid > 0) & (num_fraco > 0),           # any 'fraco'
            (num_valid > 0) & (num_medio > 0),           # any 'médio'
            (num_valid > 0) & (num_forte == num_valid),  # all 'forte'
        ]
        choices = [
            'Sinal muito fraco',
            'Sinal fraco',
            'Sinal médio',
            'Sinal forte',
        ]
        df_out['status_signal_quality'] = np.select(conditions, choices, default=None)
    else:
        df_out['status_signal_quality'] = None
        
    return df_out

def _calculate_lost_payloads_for_group(group):
    group = group.sort_values('@timestamp')
    if 'f_cnt' in group.columns:
        group['prev_f_cnt'] = group['f_cnt'].shift(1)
        lost_payload = group['f_cnt'] - group['prev_f_cnt'] - 1
        # Otimizado: .clip(lower=0) é mais rápido que .apply() para zerar negativos
        group['lost_payload'] = lost_payload.clip(lower=0).fillna(0).astype(int)
        group = group.drop(columns=['prev_f_cnt'])
    else:
        group['lost_payload'] = 0
    return group

def _calculate_rolling_metrics_for_group(group):
    group = group.sort_values('@timestamp').reset_index(drop=True)
    group_dt_index = group.set_index('@timestamp')
    if 'lost_payload' in group_dt_index.columns:
        rolling_lost = group_dt_index['lost_payload'].rolling('30D', min_periods=1).mean().round(4)
        group['lost_payloads_percent'] = rolling_lost.reset_index(drop=True)
        group['lost_payloads_status'] = np.where(group['lost_payloads_percent'] > 0.05, 'ALERT', 'OK')
        if 'f_cnt' in group.columns:
            f_cnt_alert = group['f_cnt'] == 0
            group['lost_payloads_status'] = np.where((group['lost_payloads_percent'] > 0.05) | f_cnt_alert, 'ALERT', 'OK')
    if 'optical_power_1490nm' in group.columns:
        latest_payloads = group.tail(28)
        if not latest_payloads.empty:
            base = latest_payloads['optical_power_1490nm'].iloc[0]
            if base != 0 and pd.notna(base):
                group['Var_dBm_28'] = group['optical_power_1490nm'] - base
                # Otimizado: np.where é vetorial e mais rápido que .apply()
                group['Var_dBm_28_status'] = np.where(
                    (group['Var_dBm_28'].notna()) & (group['Var_dBm_28'].abs() > 2),
                    'ALERT',
                    'OK'
                )
        group['Var_dBm_Ultimo'] = group['optical_power_1490nm'].diff()
        # Otimizado: np.where é vetorial e mais rápido que .apply()
        group['Var_dBm_Ultimo_status'] = np.where(
            (group['Var_dBm_Ultimo'].notna()) & (group['Var_dBm_Ultimo'].abs() > 2),
            'ALERT',
            'OK'
        )
    return group

def parallel_apply(df, func, group_by_col='device_id'):
    max_workers = max(1, os.cpu_count() - 2 if os.cpu_count() else 4)
    df_groups = [group for _, group in df.groupby(group_by_col)]
    if not df_groups:
        return pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, df_groups))
    return pd.concat(results)

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    print("Iniciando processamento de dados...")
    print("Carregando arquivos CSV...")
    try:
        payload_raw = pd.read_csv(config['Paths']['raw_payload'], encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        payload_raw = pd.read_csv(config['Paths']['raw_payload'], encoding='cp1252', low_memory=False)
    clientes_device_logs = pd.read_csv(config['Paths']['raw_clients'], delimiter=',', on_bad_lines='skip', low_memory=False)

    print("Realizando merge e limpeza inicial...")
    payload_raw['device_id'] = payload_raw['device_id'].astype(str).str.strip()
    clientes_device_logs['identificator_in_network'] = clientes_device_logs['identificator_in_network'].astype(str).str.strip()
    clientes_device_logs_unique = clientes_device_logs.drop_duplicates(subset=['identificator_in_network'])
    df = pd.merge(payload_raw, clientes_device_logs_unique, left_on='device_id', right_on='identificator_in_network', how='left', suffixes=('', '_log'))

    print("Tratando e renomeando colunas...")
    df['@timestamp'] = pd.to_datetime(df['@timestamp'], errors='coerce') - pd.Timedelta(hours=3)
    df['date'] = df['@timestamp'].dt.date

    prefixes = ['eyon_metadata.decoded_payload.', 'decoded_payload.', 'eyon_metadata.']
    pattern = f"^({'|'.join(map(re.escape, prefixes))})"
    new_columns = {col: re.sub(pattern, '', col) for col in df.columns}
    df.rename(columns=new_columns, inplace=True)
    df.rename(columns={'battery': 'battery_voltage'}, inplace=True, errors='ignore')

    cols = pd.Series(df.columns)
    if cols.duplicated().any():
        print("Encontradas colunas duplicadas após renomear. Consolidando...")
        for dup_name in cols[cols.duplicated()].unique():
            dup_cols = df.loc[:, df.columns == dup_name]
            consolidated_col = dup_cols.bfill(axis=1).iloc[:, 0]
            locs = np.where(df.columns == dup_name)[0]
            df.drop(columns=df.columns[locs], inplace=True)
            df.insert(locs[0], dup_name, consolidated_col)
            print(f"  - Coluna duplicada '{dup_name}' consolidada.")

    print("Limpando colunas numéricas...")
    numeric_cols = [col for col in df.columns if any(c in col for c in ['OpticalPower', 'rssi', 'rsrp', 'rsrq', 'snr', 'txPower', 'battery_voltage', 'temperature'])]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'([-+]?[0-9]*\.?[0-9]+)', expand=False), errors='coerce')

    print("Convertendo 'msg_type' para inteiro...")
    if 'msg_type' in df.columns:
        df['msg_type'] = pd.to_numeric(df['msg_type'], errors='coerce').astype('Int64')

    print("Aplicando status de sinal...")
    df = apply_signal_status_vectorized(df)

    print("Calculando payloads perdidos em paralelo...")
    df = parallel_apply(df, _calculate_lost_payloads_for_group)

    print("Calculando métricas de rolling em paralelo...")
    df = parallel_apply(df, _calculate_rolling_metrics_for_group)

    print("Salvando arquivo processado...")
    output_path = config['Paths']['processed_payload']
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Processamento concluído! {len(df):,} linhas salvas em {output_path}")



if __name__ == "__main__":
    main()