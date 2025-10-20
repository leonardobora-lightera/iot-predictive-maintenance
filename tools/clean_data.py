#!/usr/bin/env python3
"""
Data Cleaning Script for Predictive Analysis POC

This script cleans the processed dataset by:
1. Removing battery_voltage outliers (>50V or <1.5V - sensor errors)
2. Filtering other absurd values in sensor readings
3. Generating a clean dataset for training

The cleaning improves correlation between battery_voltage and failures
by removing noise from sensor malfunctions.

Usage:
    python3 tools/clean_data.py [--input FILE] [--output FILE] [--report]
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def clean_battery_voltage(df, min_voltage=1.5, max_voltage=5.0):
    """
    Clean battery_voltage outliers.
    
    Realistic IoT battery range: 1.8V (dead) to 4.2V (fully charged lithium)
    We use slightly wider range (1.5-5.0V) to be conservative.
    
    Args:
        df: DataFrame with battery_voltage column
        min_voltage: Minimum realistic voltage (default 1.5V)
        max_voltage: Maximum realistic voltage (default 5.0V)
    
    Returns:
        DataFrame with cleaned battery_voltage
    """
    initial_count = len(df)
    
    # Count outliers before cleaning
    if 'battery_voltage' in df.columns:
        outliers_low = (df['battery_voltage'] < min_voltage).sum()
        outliers_high = (df['battery_voltage'] > max_voltage).sum()
        outliers_total = outliers_low + outliers_high
        
        print(f"\n📊 Battery Voltage Analysis:")
        print(f"   Total records: {initial_count:,}")
        print(f"   Outliers <{min_voltage}V: {outliers_low:,}")
        print(f"   Outliers >{max_voltage}V: {outliers_high:,}")
        print(f"   Total outliers: {outliers_total:,} ({outliers_total/initial_count*100:.2f}%)")
        
        if outliers_total > 0:
            print(f"\n   Extreme values:")
            print(f"   Min: {df['battery_voltage'].min():.2f}V")
            print(f"   Max: {df['battery_voltage'].max():.2f}V")
            
            # Set outliers to NaN instead of removing rows
            # This preserves the record but marks battery as invalid
            df.loc[df['battery_voltage'] < min_voltage, 'battery_voltage'] = np.nan
            df.loc[df['battery_voltage'] > max_voltage, 'battery_voltage'] = np.nan
            
            print(f"\n   ✓ Outliers set to NaN (preserved records)")
            print(f"   After cleaning:")
            print(f"   Min: {df['battery_voltage'].min():.2f}V")
            print(f"   Max: {df['battery_voltage'].max():.2f}V")
    
    return df

def clean_signal_quality(df):
    """
    Clean LoRa signal quality metrics outliers.
    
    Realistic ranges:
    - RSSI: -120 to 0 dBm
    - SNR: -20 to +10 dB
    - RSRP: -140 to -40 dBm
    - RSRQ: -20 to -3 dB
    """
    outlier_count = 0
    
    if 'rssi' in df.columns:
        outliers = ((df['rssi'] < -120) | (df['rssi'] > 0)).sum()
        if outliers > 0:
            print(f"\n📡 RSSI outliers: {outliers:,}")
            df.loc[(df['rssi'] < -120) | (df['rssi'] > 0), 'rssi'] = np.nan
            outlier_count += outliers
    
    if 'snr' in df.columns:
        outliers = ((df['snr'] < -20) | (df['snr'] > 15)).sum()
        if outliers > 0:
            print(f"   SNR outliers: {outliers:,}")
            df.loc[(df['snr'] < -20) | (df['snr'] > 15), 'snr'] = np.nan
            outlier_count += outliers
    
    if 'rsrp' in df.columns:
        outliers = ((df['rsrp'] < -140) | (df['rsrp'] > -40)).sum()
        if outliers > 0:
            print(f"   RSRP outliers: {outliers:,}")
            df.loc[(df['rsrp'] < -140) | (df['rsrp'] > -40), 'rsrp'] = np.nan
            outlier_count += outliers
    
    if 'rsrq' in df.columns:
        outliers = ((df['rsrq'] < -20) | (df['rsrq'] > -3)).sum()
        if outliers > 0:
            print(f"   RSRQ outliers: {outliers:,}")
            df.loc[(df['rsrq'] < -20) | (df['rsrq'] > -3), 'rsrq'] = np.nan
            outlier_count += outliers
    
    if outlier_count > 0:
        print(f"   ✓ Total signal quality outliers cleaned: {outlier_count:,}")
    
    return df

def generate_cleaning_report(df_original, df_cleaned, output_dir='docs'):
    """Generate a report comparing original vs cleaned data."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'data_cleaning_report_{timestamp}.md')
    
    with open(report_path, 'w') as f:
        f.write("# Data Cleaning Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Original records: {len(df_original):,}\n")
        f.write(f"- Cleaned records: {len(df_cleaned):,}\n")
        f.write(f"- Records removed: {len(df_original) - len(df_cleaned):,}\n\n")
        
        f.write("## Battery Voltage Statistics\n\n")
        f.write("### Before Cleaning\n\n")
        if 'battery_voltage' in df_original.columns:
            f.write(f"```\n{df_original['battery_voltage'].describe()}\n```\n\n")
        
        f.write("### After Cleaning\n\n")
        if 'battery_voltage' in df_cleaned.columns:
            f.write(f"```\n{df_cleaned['battery_voltage'].describe()}\n```\n\n")
        
        f.write("## Correlation with Failures\n\n")
        if 'battery_voltage' in df_cleaned.columns and 'future_failure' in df_cleaned.columns:
            corr_original = df_original[['battery_voltage', 'future_failure']].corr().iloc[0, 1]
            corr_cleaned = df_cleaned[['battery_voltage', 'future_failure']].corr().iloc[0, 1]
            f.write(f"- Original correlation: {corr_original:.4f}\n")
            f.write(f"- Cleaned correlation: {corr_cleaned:.4f}\n")
            f.write(f"- Improvement: {abs(corr_cleaned) - abs(corr_original):.4f}\n\n")
        
        f.write("## Low Battery Analysis\n\n")
        if 'battery_voltage' in df_cleaned.columns and 'future_failure' in df_cleaned.columns:
            low_battery = df_cleaned[df_cleaned['battery_voltage'] < 2.8]
            f.write(f"- Samples with battery <2.8V: {len(low_battery):,}\n")
            if len(low_battery) > 0:
                f.write(f"- Failure rate in low battery: {low_battery['future_failure'].mean()*100:.2f}%\n")
            f.write(f"- Overall failure rate: {df_cleaned['future_failure'].mean()*100:.2f}%\n\n")
    
    print(f"\n✓ Cleaning report saved to {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Clean processed dataset for better training')
    parser.add_argument('--input', type=str, default='data/processed/payloads_processed.csv',
                        help='Input CSV file (default: data/processed/payloads_processed.csv)')
    parser.add_argument('--output', type=str, default='data/processed/payloads_processed_clean.csv',
                        help='Output cleaned CSV file (default: data/processed/payloads_processed_clean.csv)')
    parser.add_argument('--report', action='store_true',
                        help='Generate cleaning report in docs/')
    parser.add_argument('--min-battery', type=float, default=1.5,
                        help='Minimum realistic battery voltage (default: 1.5V)')
    parser.add_argument('--max-battery', type=float, default=5.0,
                        help='Maximum realistic battery voltage (default: 5.0V)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DATA CLEANING FOR PREDICTIVE ANALYSIS")
    print("=" * 70)
    
    # Load data
    print(f"\n1. Loading data from {args.input}...")
    try:
        df_original = pd.read_csv(args.input, low_memory=False)
        print(f"   ✓ Loaded {len(df_original):,} records")
        print(f"   ✓ Columns: {len(df_original.columns)}")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return 1
    
    # Create a copy for cleaning
    df_cleaned = df_original.copy()
    
    # Clean battery voltage
    print(f"\n2. Cleaning battery_voltage...")
    df_cleaned = clean_battery_voltage(df_cleaned, args.min_battery, args.max_battery)
    
    # Clean signal quality metrics
    print(f"\n3. Cleaning signal quality metrics...")
    df_cleaned = clean_signal_quality(df_cleaned)
    
    # Save cleaned data
    print(f"\n4. Saving cleaned data to {args.output}...")
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df_cleaned.to_csv(args.output, index=False)
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"   ✓ Saved {len(df_cleaned):,} records ({file_size_mb:.2f} MB)")
    except Exception as e:
        print(f"   ✗ Error saving data: {e}")
        return 1
    
    # Generate report if requested
    if args.report:
        print(f"\n5. Generating cleaning report...")
        generate_cleaning_report(df_original, df_cleaned)
    
    print("\n" + "=" * 70)
    print("CLEANING COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review the cleaned data: {args.output}")
    if args.report:
        print(f"2. Check the report in docs/")
    print(f"3. Train a new model using the clean data:")
    print(f"   python3 tools/train_monitor.py --all --temporal-split --preserve-model")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
