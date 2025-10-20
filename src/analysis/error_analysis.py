import pandas as pd
import configparser


def main():
    """
    Analyzes the processed data to count the occurrences of each error code.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    processed_payload_path = config['Paths']['processed_payload']

    try:
        df = pd.read_csv(processed_payload_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file {processed_payload_path} was not found.")
        return

    error_cols = [col for col in df.columns if col.startswith('buffer.error_codes.')]

    if not error_cols:
        print("No error code columns found in the data.")
        return

    all_errors = df[error_cols].stack().dropna()
    all_errors = all_errors[all_errors != 0]

    if all_errors.empty:
        print("No errors found in the data.")
        return

    error_counts = all_errors.value_counts()

    print("--- Error Code Analysis ---")
    print(error_counts)


if __name__ == "__main__":
    main()
