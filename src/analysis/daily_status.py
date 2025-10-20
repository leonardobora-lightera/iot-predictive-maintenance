import os
import glob
import pandas as pd
from datetime import datetime

def process_daily_status_from_backups():
    backup_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backups')
    file_pattern = os.path.join(backup_dir, 'clientes_device_logs_*.csv')
    backup_files = glob.glob(file_pattern)
    
    live_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'clientes_device_logs.csv')
    if os.path.exists(live_file_path):
        all_files = backup_files + [live_file_path]
    else:
        all_files = backup_files

    if not all_files:
        return pd.DataFrame(columns=['date'])

    all_statuses = set()
    # First pass: get all unique statuses
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            if 'network_server' in df.columns and 'status' in df.columns:
                nbiot_df = df[df['network_server'] == 'NBIoT']
                all_statuses.update(nbiot_df['status'].dropna().unique())
        except Exception as e:
            print(f"Error reading statuses from file {file_path}: {e}")
            continue

    daily_counts = []
    processed_dates = set()

    # Second pass: process files and count statuses
    for file_path in all_files:
        try:
            if 'backups' in file_path:
                date_str = os.path.basename(file_path).replace('clientes_device_logs_', '').replace('.csv', '')
                file_date = pd.to_datetime(date_str).date()
            else: # live file
                file_date = datetime.now().date()

            if file_date in processed_dates:
                continue
            processed_dates.add(file_date)

            df = pd.read_csv(file_path)

            if 'network_server' in df.columns and 'status' in df.columns:
                nbiot_df = df[df['network_server'] == 'NBIoT']
                status_counts = nbiot_df['status'].value_counts()

                daily_data = {'date': file_date}
                for status in all_statuses:
                    daily_data[status] = status_counts.get(status, 0)
                
                daily_counts.append(daily_data)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if not daily_counts:
        return pd.DataFrame(columns=['date'] + list(all_statuses))

    result_df = pd.DataFrame(daily_counts)
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df = result_df.sort_values(by='date').reset_index(drop=True)
    
    # Ensure all status columns are present, even if they had no counts in any file
    for status in all_statuses:
        if status not in result_df.columns:
            result_df[status] = 0
            
    return result_df
