#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for Phase 1 Model

This script loads the Phase 1 model and computes SHAP feature importance
to understand which features contribute most to failure predictions.

Usage:
    python3 tools/shap_analysis.py [--model-path MODEL] [--output-dir DIR] [--samples N]
"""

import argparse
import sys
import os
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.logic import load_data
from src.collector.train_model import feature_engineering

def analyze_feature_importance(model_path, output_dir, n_samples=1000):
    """
    Compute SHAP feature importance for the trained model.
    
    Args:
        model_path: Path to the saved model.joblib
        output_dir: Directory to save analysis outputs
        n_samples: Number of samples to use for SHAP analysis (default 1000)
    """
    print("=" * 70)
    print("SHAP Feature Importance Analysis")
    print("=" * 70)
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    try:
        model_payload = joblib.load(model_path)
        model = model_payload['model']
        features = model_payload['features']
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Features: {len(features)} features")
        print(f"   ✓ Feature list: {features}")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return None
    
    # Load and prepare data
    print(f"\n2. Loading and preparing data...")
    try:
        data = load_data()
        if data is None or data.empty:
            print("   ✗ Failed to load data")
            return None
        
        print(f"   ✓ Data loaded: {len(data)} rows")
        
        # Use last 90 days for consistency with training
        data['@timestamp'] = pd.to_datetime(data['@timestamp'])
        data_90d = data[data['@timestamp'] >= data['@timestamp'].max() - pd.Timedelta(days=90)]
        print(f"   ✓ Filtered to 90-day window: {len(data_90d)} rows")
        
        # Feature engineering
        df_features = feature_engineering(data_90d)
        print(f"   ✓ Feature engineering complete: {len(df_features)} rows")
        
        # Filter to valid samples with all features
        valid_samples = df_features.dropna(subset=features + ['future_failure'])
        print(f"   ✓ Valid samples (no NaN): {len(valid_samples)} rows")
        
        if len(valid_samples) == 0:
            print("   ✗ No valid samples after filtering")
            return None
            
    except Exception as e:
        print(f"   ✗ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Sample data for SHAP (computationally expensive on large datasets)
    print(f"\n3. Sampling {n_samples} rows for SHAP analysis...")
    if len(valid_samples) > n_samples:
        sample_data = valid_samples.sample(n=n_samples, random_state=42)
        print(f"   ✓ Sampled {len(sample_data)} rows")
    else:
        sample_data = valid_samples
        print(f"   ✓ Using all {len(sample_data)} rows (less than requested sample size)")
    
    X_sample = sample_data[features]
    y_sample = sample_data['future_failure']
    
    print(f"   ✓ Sample distribution: {y_sample.value_counts().to_dict()}")
    
    # Compute SHAP values
    print(f"\n4. Computing SHAP values...")
    try:
        import shap
        
        # Use TreeExplainer for RandomForest
        explainer = shap.TreeExplainer(model)
        print(f"   ✓ TreeExplainer created")
        
        # Compute SHAP values
        print(f"   ⏳ Computing SHAP values (this may take a minute)...")
        shap_values = explainer.shap_values(X_sample)
        print(f"   ✓ SHAP values computed")
        
        # Handle multi-output (binary classification returns list of arrays)
        if isinstance(shap_values, list):
            # Use class 1 (failure) SHAP values
            shap_values_class1 = shap_values[1]
            print(f"   ℹ Using SHAP values for class 1 (failure) from list")
        elif len(shap_values.shape) == 3:
            # Shape is (n_samples, n_features, n_classes)
            # Extract class 1 (failure) from last dimension
            shap_values_class1 = shap_values[:, :, 1]
            print(f"   ℹ Using SHAP values for class 1 (failure) from 3D array")
        else:
            shap_values_class1 = shap_values
            print(f"   ℹ Using SHAP values directly")
        
        print(f"   ✓ SHAP values shape (after extraction): {shap_values_class1.shape}")
        
    except Exception as e:
        print(f"   ✗ Error computing SHAP values: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Analyze feature importance
    print(f"\n5. Analyzing feature importance...")
    
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': shap_values_class1.mean(axis=0),
        'std_shap': shap_values_class1.std(axis=0)
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    
    # Calculate percentage contribution
    total_importance = importance_df['mean_abs_shap'].sum()
    importance_df['pct_contribution'] = (importance_df['mean_abs_shap'] / total_importance) * 100
    importance_df['cumulative_pct'] = importance_df['pct_contribution'].cumsum()
    
    print("\n" + "=" * 70)
    print("Feature Importance Ranking (by mean |SHAP value|)")
    print("=" * 70)
    print(importance_df.to_string(index=False))
    print("=" * 70)
    
    # Identify low-impact features (<1% contribution)
    low_impact = importance_df[importance_df['pct_contribution'] < 1.0]
    if len(low_impact) > 0:
        print(f"\n⚠️  Low-impact features (<1% contribution): {len(low_impact)}")
        print(low_impact[['feature', 'pct_contribution']].to_string(index=False))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save importance CSV
    csv_path = os.path.join(output_dir, f'feature_importance_{timestamp}.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"\n✓ Feature importance saved to {csv_path}")
    
    # Save JSON report
    report = {
        'timestamp': timestamp,
        'model_path': model_path,
        'n_samples': len(sample_data),
        'n_features': len(features),
        'features': features,
        'importance_ranking': importance_df.to_dict('records'),
        'summary': {
            'top_feature': importance_df.iloc[0]['feature'],
            'top_feature_contribution_pct': float(importance_df.iloc[0]['pct_contribution']),
            'top_3_cumulative_pct': float(importance_df.iloc[:3]['pct_contribution'].sum()),
            'low_impact_features': low_impact['feature'].tolist() if len(low_impact) > 0 else []
        }
    }
    
    json_path = os.path.join(output_dir, f'shap_analysis_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Analysis report saved to {json_path}")
    
    # Save SHAP values for later visualization
    shap_data = {
        'shap_values': shap_values_class1,
        'X_sample': X_sample,
        'features': features,
        'expected_value': explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    }
    shap_pkl_path = os.path.join(output_dir, f'shap_values_{timestamp}.pkl')
    joblib.dump(shap_data, shap_pkl_path)
    print(f"✓ SHAP values saved to {shap_pkl_path}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return importance_df

def main():
    parser = argparse.ArgumentParser(description='Analyze SHAP feature importance for trained model')
    parser.add_argument('--model-path', type=str, default='model.joblib', 
                        help='Path to model.joblib (default: model.joblib)')
    parser.add_argument('--output-dir', type=str, default='shap_analysis_output',
                        help='Directory to save outputs (default: shap_analysis_output)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples for SHAP analysis (default: 1000)')
    args = parser.parse_args()
    
    result = analyze_feature_importance(args.model_path, args.output_dir, args.samples)
    
    if result is not None:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
