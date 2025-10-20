"""
This script identifies devices that have not sent a payload for more than a specified period (default is 24 hours).
"""

import sys
import os
import pandas as pd
from datetime import timedelta

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.logic import load_data

def detect_communication_failures(threshold_hours=24):
    """
    Detects and reports devices that have been silent for longer than the specified threshold.

    Args:
        threshold_hours (int): The number of hours to consider a device silent.
    """
    print("Running communication failure detection...")
    
    try:
        df = load_data()
        if df.empty:
            print("Dataframe is empty. Cannot perform analysis.")
            return
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Ensure timestamp column is in datetime format and timezone-aware (UTC)
    if '@timestamp' not in df.columns:
        print("Error: '@timestamp' column not found in the data.")
        return
        
    df['@timestamp'] = pd.to_datetime(df['@timestamp'], utc=True)

    # Get the current time in UTC to ensure a correct comparison
    now = pd.to_datetime('now', utc=True)
    
    # Find the last communication timestamp for each device
    last_communication = df.groupby('device_id')['@timestamp'].max()
    
    # Calculate the time elapsed since the last communication
    time_since_last_comm = now - last_communication
    
    # Define the failure threshold
    failure_threshold = timedelta(hours=threshold_hours)
    
    # Identify silent devices
    silent_devices = time_since_last_comm[time_since_last_comm > failure_threshold]
    
    print("-" * 50)
    if not silent_devices.empty:
        print(f"Found {len(silent_devices)} devices that have not communicated in over {threshold_hours} hours:")
        print("-" * 50)
        for device_id, time_diff in silent_devices.items():
            days = time_diff.days
            hours, remainder = divmod(time_diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            print(f"  - Device ID: {device_id}, Last seen: {days} days and {hours} hours ago")
    else:
        print(f"Success: All devices have communicated within the last {threshold_hours} hours.")
    print("-" * 50)

if __name__ == "__main__":
    detect_communication_failures()