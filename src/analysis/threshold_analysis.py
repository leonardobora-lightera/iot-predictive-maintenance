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
    """
    Analyzes the processed data to flag instances where thresholds are exceeded.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    processed_payload_path = config['Paths']['processed_payload']

    try:
        df = pd.read_csv(processed_payload_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file {processed_payload_path} was not found.")
        return

    print("--- Threshold Analysis ---")

    for device_id, group in df.groupby('device_id'):
        print(f"\nDevice: {device_id}")

        if 'Lux' in group.columns:
            lux_violations = group[group['Lux'] > THRESHOLDS['luminosity']]
            if not lux_violations.empty:
                print(f"  Luminosity threshold exceeded: {len(lux_violations)} times")

        if 'Pitch_Acelerometro' in group.columns:
            pitch_violations = group[group['Pitch_Acelerometro'].abs() > THRESHOLDS['accelerometer']]
            if not pitch_violations.empty:
                print(f"  Accelerometer (Pitch) threshold exceeded: {len(pitch_violations)} times")

        if 'Roll_Acelerometro' in group.columns:
            roll_violations = group[group['Roll_Acelerometro'].abs() > THRESHOLDS['accelerometer']]
            if not roll_violations.empty:
                print(f"  Accelerometer (Roll) threshold exceeded: {len(roll_violations)} times")

        if 'Potencia_Optica_1490nm' in group.columns:
            power_violations = group[group['Potencia_Optica_1490nm'] < THRESHOLDS['powermeter']]
            if not power_violations.empty:
                print(f"  Powermeter threshold exceeded: {len(power_violations)} times")

        if 'Temperatura_C' in group.columns:
            temp_violations = group[group['Temperatura_C'] > THRESHOLDS['temperature']]
            if not temp_violations.empty:
                print(f"  Temperature threshold exceeded: {len(temp_violations)} times")

        if 'Bateria' in group.columns:
            battery_violations = group[group['Bateria'] < THRESHOLDS['battery']]
            if not battery_violations.empty:
                print(f"  Battery threshold exceeded: {len(battery_violations)} times")

if __name__ == "__main__":
    main()
