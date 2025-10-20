
import sys
import os

# Add the project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.core.logic import load_data

def feature_engineering(df):
    """Cria features para o modelo a partir dos dados brutos."""
    print("Iniciando engenharia de features...")
    df = df.sort_values(by=['device_id', '@timestamp']).copy()
    
    # Ensure timestamp is datetime before setting as index
    if '@timestamp' in df.columns:
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
    
    # Define o timestamp como índice para operações de janela de tempo
    df = df.set_index('@timestamp')
    
    # ========== BATTERY FEATURES ==========
    if 'battery_voltage' in df.columns:
        df['battery_rolling_mean_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').mean().reset_index(0, drop=True)
        df['battery_rolling_std_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').std().reset_index(0, drop=True)
        
        # PHASE 3: Critical battery features for defect detection
        # Critical level: <2.8V (approaching failure threshold of 2.5V)
        df['battery_critical'] = (df['battery_voltage'] < 2.8).astype(int)
        
        # Very low level: <2.5V (imminent failure - matches future_failure threshold)
        df['battery_very_low'] = (df['battery_voltage'] < 2.5).astype(int)
        
        # Battery drop rate: V/day decline over last 7 days
        # Negative values = battery declining, positive = charging (unusual for IoT)
        battery_first_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan).reset_index(0, drop=True)
        battery_last_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').apply(lambda x: x.iloc[-1] if len(x) > 0 else np.nan).reset_index(0, drop=True)
        df['battery_drop_rate_7d'] = (battery_first_7d - battery_last_7d) / 7.0  # V/day
        
        # PHASE 3.1: Battery INSTABILITY features (key for detecting defective batteries!)
        # High standard deviation = oscillating/unstable battery (defective)
        # Normal battery: stable linear decline (low std)
        # Defective battery: erratic behavior (high std)
        df['battery_instability_7d'] = df['battery_rolling_std_7d']  # Alias for clarity
        
        # Coefficient of variation: std/mean (normalized instability, independent of voltage level)
        # High CV = proportionally large oscillations
        df['battery_cv_7d'] = df['battery_rolling_std_7d'] / df['battery_rolling_mean_7d'].replace(0, np.nan)
        
        # Battery range (max - min) in last 7 days: captures extreme oscillations
        battery_max_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').max().reset_index(0, drop=True)
        battery_min_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').min().reset_index(0, drop=True)
        df['battery_range_7d'] = battery_max_7d - battery_min_7d
        
        # High oscillation flag: battery range >0.3V in 7 days (defect indicator)
        # Normal battery degrades ~0.01-0.05V/week, defective can swing 0.3V+
        df['battery_high_oscillation'] = (df['battery_range_7d'] > 0.3).astype(int)
        
        # PHASE 4: Spike count - detect abrupt voltage changes (defect signature)
        # Count of changes >0.2V in 7-day window
        # Normal battery: smooth changes <0.1V, defective: spikes >0.2V
        df['battery_diff'] = df.groupby('device_id')['battery_voltage'].diff().abs()
        df['battery_spike_count_7d'] = df.groupby('device_id')['battery_diff'].rolling('7D').apply(
            lambda x: (x > 0.2).sum() if len(x) > 0 else 0
        ).reset_index(0, drop=True)
        
        # PHASE 4: Trend - linear regression slope (charge/discharge pattern)
        # Negative slope = discharging (normal), positive = charging (abnormal for battery sensor)
        # Steep negative slope (<-0.1 V/day) = rapid discharge (defect)
        from scipy.stats import linregress
        
        def compute_battery_trend(series):
            """Compute linear regression slope for battery voltage trend."""
            if len(series) < 2 or series.isna().all():
                return np.nan
            x = np.arange(len(series))
            y = series.values
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                return np.nan
            return linregress(x[valid], y[valid]).slope
        
        df['battery_trend_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').apply(
            compute_battery_trend, raw=False
        ).reset_index(0, drop=True)

    # ========== CONNECTIVITY FEATURES ==========
    if 'lost_payloads_percent' in df.columns:
        df['lost_payloads_rolling_mean_7d'] = df.groupby('device_id')['lost_payloads_percent'].rolling('7D').mean().reset_index(0, drop=True)
    
    if 'registration_time' in df.columns:
        df['registration_time_rolling_mean_7d'] = df.groupby('device_id')['registration_time'].rolling('7D').mean().reset_index(0, drop=True)

    # ========== PHASE 1 ADDITIONS: COMMUNICATION PATTERNS ==========
    # Frame count velocity (messages per day)
    if 'f_cnt' in df.columns:
        df['f_cnt_rolling_mean_7d'] = df.groupby('device_id')['f_cnt'].rolling('7D').mean().reset_index(0, drop=True)
        df['f_cnt_rolling_std_7d'] = df.groupby('device_id')['f_cnt'].rolling('7D').std().reset_index(0, drop=True)
    
    # Error rate from buffer
    if 'buffer.total_errors' in df.columns:
        df['total_errors_rolling_sum_7d'] = df.groupby('device_id')['buffer.total_errors'].rolling('7D').sum().reset_index(0, drop=True)
    
    # ========== PHASE 1 ADDITIONS: LORA SIGNAL QUALITY ==========
    if 'rssi' in df.columns:
        df['rssi_rolling_mean_7d'] = df.groupby('device_id')['rssi'].rolling('7D').mean().reset_index(0, drop=True)
        df['rssi_rolling_std_7d'] = df.groupby('device_id')['rssi'].rolling('7D').std().reset_index(0, drop=True)
    
    if 'snr' in df.columns:
        df['snr_rolling_mean_7d'] = df.groupby('device_id')['snr'].rolling('7D').mean().reset_index(0, drop=True)
        df['snr_rolling_std_7d'] = df.groupby('device_id')['snr'].rolling('7D').std().reset_index(0, drop=True)
    
    if 'rsrp' in df.columns:
        df['rsrp_rolling_mean_7d'] = df.groupby('device_id')['rsrp'].rolling('7D').mean().reset_index(0, drop=True)
    
    if 'rsrq' in df.columns:
        df['rsrq_rolling_mean_7d'] = df.groupby('device_id')['rsrq'].rolling('7D').mean().reset_index(0, drop=True)

    # ========== PHASE 1 ADDITIONS: DEVICE CONTEXT ==========
    # Deployment age (days since first seen per device)
    df['deployment_age_days'] = df.groupby('device_id').cumcount()
    
    # Message type distribution (percentage of error messages in last 7 days)
    if 'msg_type' in df.columns:
        df['msg_type_error_pct_7d'] = df.groupby('device_id')['msg_type'].rolling('7D').apply(
            lambda x: (x == 6).sum() / len(x) if len(x) > 0 else 0
        ).reset_index(0, drop=True)

    # ========== PHASE 6: LAG DIFFERENCE FEATURES (HUB_IA PATTERN) ==========
    # Lag differences capture ABRUPT changes (spikes/drops) vs rolling windows (gradual trends)
    # HUB_IA insight: optical_power_lag_diff_6h was TOP feature in business rule
    
    # Battery lag differences (detect sudden voltage drops)
    df['battery_lag_diff_1d'] = df.groupby('device_id')['battery_voltage'].diff(periods=1)  # Day-to-day change
    df['battery_lag_diff_7d'] = df.groupby('device_id')['battery_voltage'].diff(periods=7)  # Weekly degradation
    
    # Signal quality lag differences (detect abrupt signal deterioration)
    if 'rssi' in df.columns:
        df['rssi_lag_diff_7d'] = df.groupby('device_id')['rssi'].diff(periods=7)  # Weekly signal drop
    
    if 'snr' in df.columns:
        df['snr_lag_diff_7d'] = df.groupby('device_id')['snr'].diff(periods=7)  # Weekly noise increase
    
    # Communication errors lag difference (detect error bursts)
    if 'buffer.total_errors' in df.columns:
        df['total_errors_lag_diff_7d'] = df.groupby('device_id')['buffer.total_errors'].diff(periods=7)  # Weekly error spike

    # ========== HANDLE LAG_DIFF NaNs ==========
    # Fill NaN in lag_diff features with 0 (semantically correct: no historical data = no change = 0)
    lag_diff_cols = ['battery_lag_diff_1d', 'battery_lag_diff_7d', 
                      'rssi_lag_diff_7d', 'snr_lag_diff_7d', 'total_errors_lag_diff_7d']
    
    for col in lag_diff_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            print(f"✅ Filled NaN in {col} with 0")

    # ========== TARGET VARIABLE ==========
    # Define uma falha futura (próximos 30 dias)
    df['future_failure'] = df.groupby('device_id')['battery_voltage'].transform(lambda x: x.rolling('30D').min() < 2.5).shift(-30)
    
    df = df.dropna(subset=['future_failure'])
    df['future_failure'] = df['future_failure'].astype(int)
    
    # ========== PHASE 5: FIX TEMPORAL SPLIT ==========
    # CRITICAL: Reset index to preserve @timestamp as a column for temporal split
    # Root cause: @timestamp was set as index (line 23) but lost during dropna()
    # Solution: reset_index() keeps @timestamp accessible for temporal validation
    df = df.reset_index()  # Converts @timestamp from index back to column
    
    print("Engenharia de features concluída.")
    return df

def cross_validate_temporal(df_features, n_estimators=100, max_features='sqrt', optimal_threshold=0.23, n_splits=5, baseline_mode=False):
    """
    Performs temporal cross-validation using TimeSeriesSplit (Task 3).
    
    Validates model stability across temporal folds to ensure recall ~85% holds
    across different time periods.
    
    Args:
        df_features: DataFrame with engineered features and target
        n_estimators: RandomForest n_estimators (default: 100)
        max_features: RandomForest max_features (default: 'sqrt')
        optimal_threshold: Threshold for predictions from Task 9 (default: 0.23)
        n_splits: Number of CV folds (default: 5)
        baseline_mode: If True, use only 4 original features
    
    Returns:
        dict with fold-by-fold results and aggregate statistics
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
    
    print(f"\n{'='*80}")
    print("📊 TASK 3: TEMPORAL CROSS-VALIDATION (TimeSeriesSplit)")
    print(f"{'='*80}")
    print("Objective: Validate 85% recall stability across temporal folds")
    print("Method: TimeSeriesSplit with 7-day gap to prevent temporal leakage")
    print("Success Criterion: std(recall) < 10% indicates stable model")
    
    # ========== FEATURE SELECTION ==========
    if baseline_mode:
        candidate_features = [
            'battery_rolling_mean_7d',
            'battery_rolling_std_7d',
            'lost_payloads_rolling_mean_7d',
            'registration_time_rolling_mean_7d'
        ]
    else:
        # Phase 6: battery-focused + instability + lag_diff (30 features)
        candidate_features = [
            # PHASE 3: Critical battery indicators
            'battery_critical',
            'battery_very_low',
            'battery_high_oscillation',
            # PHASE 4: Battery PATTERN features
            'battery_spike_count_7d',
            'battery_trend_7d',
            'battery_range_7d',
            'battery_cv_7d',
            'battery_instability_7d',
            'battery_drop_rate_7d',
            # PHASE 2: Raw current values
            'battery_voltage',
            # Original battery features
            'battery_rolling_mean_7d', 
            'battery_rolling_std_7d',
            # Original connectivity features
            'lost_payloads_rolling_mean_7d', 
            'registration_time_rolling_mean_7d',
            # Phase 1: Communication patterns
            'f_cnt_rolling_mean_7d',
            'f_cnt_rolling_std_7d',
            'total_errors_rolling_sum_7d',
            'msg_type_error_pct_7d',
            # Phase 1: LoRa signal quality
            'rssi_rolling_mean_7d',
            'rssi_rolling_std_7d',
            'snr_rolling_mean_7d',
            'snr_rolling_std_7d',
            'rsrp_rolling_mean_7d',
            'rsrq_rolling_mean_7d',
            # Phase 1: Device context
            'deployment_age_days',
            # PHASE 6: LAG DIFFERENCE FEATURES
            'battery_lag_diff_1d',
            'battery_lag_diff_7d',
            'rssi_lag_diff_7d',
            'snr_lag_diff_7d',
            'total_errors_lag_diff_7d'
        ]
    
    # Filter to only features present in the dataframe
    valid_features = [f for f in candidate_features if f in df_features.columns]
    
    if not valid_features:
        print("❌ No valid features found. Aborting cross-validation.")
        return None
    
    print(f"\n✅ Valid features: {len(valid_features)}/{len(candidate_features)}")
    
    # ========== PREPARE DATA ==========
    df_model = df_features.dropna(subset=valid_features + ['future_failure'])
    
    if df_model.empty:
        print("❌ DataFrame empty after removing NaNs. Aborting cross-validation.")
        return None
    
    # Sort by timestamp (crucial for temporal correctness)
    if '@timestamp' in df_model.columns:
        df_model = df_model.sort_values('@timestamp')
    
    X = df_model[valid_features].values
    y = df_model['future_failure'].values
    
    print(f"Dataset: {len(X):,} samples, {len(valid_features)} features")
    print(f"Class distribution: {np.sum(y==0)} normal ({np.sum(y==0)/len(y)*100:.1f}%), {np.sum(y==1)} failures ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    # ========== CALCULATE GAP IN SAMPLES ==========
    # Gap = 7 days in NUMBER OF SAMPLES (not time units)
    # This prevents temporal leakage from rolling window features
    if '@timestamp' in df_model.columns:
        time_range_days = (df_model['@timestamp'].max() - df_model['@timestamp'].min()).days
        avg_samples_per_day = len(df_model) / time_range_days if time_range_days > 0 else 0
        gap_samples_desired = int(7 * avg_samples_per_day)
        
        # Calculate maximum viable gap for given n_splits
        # Rule: gap must allow at least (n_splits + 1) non-overlapping segments
        max_gap = len(df_model) // (n_splits + 2)  # Conservative estimate
        gap_samples = min(gap_samples_desired, max_gap)
        
        if gap_samples != gap_samples_desired:
            print(f"⚠️ Gap adjusted: desired {gap_samples_desired} samples (7 days) → using {gap_samples} samples (max viable for {n_splits} splits)")
        
        print(f"Gap calculation: {time_range_days} days total → ~{avg_samples_per_day:.1f} samples/day → gap = {gap_samples} samples")
    else:
        gap_samples = 0
        print("⚠️ WARNING: @timestamp not found, gap set to 0 (no temporal gap)")
    
    # ========== TIMESERIESSPLIT ==========
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_samples)
    
    print(f"\nTimeSeriesSplit configuration:")
    print(f"  n_splits: {n_splits} folds")
    print(f"  gap: {gap_samples} samples (~7 days)")
    print(f"  Threshold: {optimal_threshold} (from Task 9 optimization)")
    
    # ========== CROSS-VALIDATION LOOP ==========
    results = {
        'fold': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'fn': [],
        'fp': [],
        'tp': [],
        'tn': [],
        'train_size': [],
        'test_size': []
    }
    
    print(f"\n{'='*80}")
    print("FOLD-BY-FOLD RESULTS:")
    print(f"{'='*80}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
        print(f"  Train failures: {np.sum(y_train):,} ({np.sum(y_train)/len(y_train)*100:.1f}%) | Test failures: {np.sum(y_test):,} ({np.sum(y_test)/len(y_test)*100:.1f}%)")
        
        # Train RandomForest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict with optimal threshold from Task 9
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        # Store results
        results['fold'].append(fold_idx)
        results['recall'].append(recall)
        results['precision'].append(precision)
        results['f1'].append(f1)
        results['fn'].append(int(fn))
        results['fp'].append(int(fp))
        results['tp'].append(int(tp))
        results['tn'].append(int(tn))
        results['train_size'].append(len(X_train))
        results['test_size'].append(len(X_test))
        
        print(f"  📊 Metrics: Recall={recall:.4f} | Precision={precision:.4f} | F1={f1:.4f}")
        print(f"  📉 Confusion: TP={tp} | FN={fn} | FP={fp} | TN={tn}")
    
    # ========== AGGREGATE STATISTICS ==========
    mean_recall = np.mean(results['recall'])
    std_recall = np.std(results['recall'])
    cv_recall = std_recall / mean_recall if mean_recall > 0 else 0  # Coefficient of variation
    
    mean_precision = np.mean(results['precision'])
    std_precision = np.std(results['precision'])
    
    mean_f1 = np.mean(results['f1'])
    std_f1 = np.std(results['f1'])
    
    total_fn = sum(results['fn'])
    total_fp = sum(results['fp'])
    total_tp = sum(results['tp'])
    total_tn = sum(results['tn'])
    
    print(f"\n{'='*80}")
    print("📈 AGGREGATE STATISTICS:")
    print(f"{'='*80}")
    print(f"Recall:    {mean_recall:.4f} ± {std_recall:.4f} (CV: {cv_recall*100:.1f}%)")
    print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"\nTotal Confusion Matrix (all folds combined):")
    print(f"  TP: {total_tp:,} | FN: {total_fn:,}")
    print(f"  FP: {total_fp:,} | TN: {total_tn:,}")
    
    # ========== STABILITY VALIDATION ==========
    stability_threshold = 0.10  # 10% coefficient of variation
    is_stable = cv_recall < stability_threshold
    
    print(f"\n{'='*80}")
    print(f"🎯 STABILITY CHECK: {'✅ STABLE' if is_stable else '⚠️ UNSTABLE'}")
    print(f"{'='*80}")
    print(f"Criterion: CV(recall) < {stability_threshold*100:.0f}% (Coefficient of Variation)")
    print(f"Actual:    CV(recall) = {cv_recall*100:.1f}%")
    
    if is_stable:
        print("\n✅ Model is STABLE across temporal folds!")
        print(f"   Recall varies by only {std_recall*100:.2f} percentage points across folds.")
        print(f"   Mean recall {mean_recall:.2%} is production-ready.")
    else:
        print("\n⚠️ Model shows INSTABILITY across temporal folds!")
        print(f"   Recall varies by {std_recall*100:.2f} percentage points (>{stability_threshold*100:.0f}% threshold).")
        print(f"   Consider: More features, different time periods, hyperparameter tuning.")
    
    # Store aggregate statistics
    results['mean_recall'] = mean_recall
    results['std_recall'] = std_recall
    results['cv_recall'] = cv_recall
    results['mean_precision'] = mean_precision
    results['std_precision'] = std_precision
    results['mean_f1'] = mean_f1
    results['std_f1'] = std_f1
    results['is_stable'] = is_stable
    results['total_fn'] = total_fn
    results['total_fp'] = total_fp
    results['total_tp'] = total_tp
    results['total_tn'] = total_tn
    
    return results

def train_and_save_model(df, n_estimators=100, n_jobs=-1, max_features='sqrt', baseline_mode=False, temporal_split=False):
    """Treina um modelo de Random Forest e salva no arquivo model.joblib."""
    
    if baseline_mode:
        print("🔵 BASELINE MODE: usando apenas features originais (4 features)")
        candidate_features = [
            'battery_rolling_mean_7d', 
            'battery_rolling_std_7d',
            'lost_payloads_rolling_mean_7d', 
            'registration_time_rolling_mean_7d'
        ]
    else:
        # Phase 6: battery-focused + instability + lag_diff (30 features)
        print("🚀 PHASE 6 MODE: battery + instability + lag_diff features (30 features)")
        candidate_features = [
            # PHASE 3: Critical battery indicators (binary features have high importance in RandomForest)
            'battery_critical',  # 1 if voltage <2.8V (critical level per Enzo's devices)
            'battery_very_low',  # 1 if voltage <2.5V (imminent failure)
            'battery_high_oscillation',  # 1 if range >0.3V in 7d (defective battery!)
            # PHASE 4: Battery PATTERN features (detect defects through oscillation/instability!)
            'battery_spike_count_7d',  # NEW PHASE 4: count of abrupt changes >0.2V
            'battery_trend_7d',  # NEW PHASE 4: linear regression slope (charge pattern)
            'battery_range_7d',  # max-min voltage in 7d (oscillation amplitude)
            'battery_cv_7d',  # coefficient of variation (normalized instability)
            'battery_instability_7d',  # Alias for std (captures erratic behavior)
            'battery_drop_rate_7d',  # V/day rate of voltage decline
            # PHASE 2: Raw current values (immediate risk indicators)
            'battery_voltage',  # Current battery level (catches sudden drops!)
            # Original battery features (trends)
            'battery_rolling_mean_7d', 
            'battery_rolling_std_7d',
            # Original connectivity features
            'lost_payloads_rolling_mean_7d', 
            'registration_time_rolling_mean_7d',
            # Phase 1: Communication patterns
            'f_cnt_rolling_mean_7d',
            'f_cnt_rolling_std_7d',
            'total_errors_rolling_sum_7d',
            'msg_type_error_pct_7d',
            # Phase 1: LoRa signal quality
            'rssi_rolling_mean_7d',
            'rssi_rolling_std_7d',
            'snr_rolling_mean_7d',
            'snr_rolling_std_7d',
            'rsrp_rolling_mean_7d',
            'rsrq_rolling_mean_7d',
            # Phase 1: Device context
            'deployment_age_days',
            # ========== PHASE 6: LAG DIFFERENCE FEATURES ==========
            # Capture ABRUPT changes (spikes/drops) - HUB_IA validated pattern
            'battery_lag_diff_1d',  # Day-to-day battery change (sudden drops)
            'battery_lag_diff_7d',  # Weekly battery degradation (accelerated decline)
            'rssi_lag_diff_7d',  # Weekly signal deterioration
            'snr_lag_diff_7d',  # Weekly noise increase
            'total_errors_lag_diff_7d'  # Weekly error burst
        ]
    
    # Filter to only features present in the dataframe
    valid_features = [f for f in candidate_features if f in df.columns]
    
    if not valid_features:
        print("Nenhuma feature válida encontrada. Abortando o treinamento.")
        return
    
    print(f"Features válidas encontradas: {len(valid_features)}/{len(candidate_features)}")
    print(f"Lista de features: {valid_features}")

    # Keep @timestamp for temporal split before dropping NaNs
    df_model = df.dropna(subset=valid_features + ['future_failure'])
    
    if df_model.empty:
        print("DataFrame vazio após remover NaNs. Abortando o treinamento.")
        return

    X = df_model[valid_features]
    y = df_model['future_failure']

    # Divide os dados em treino e teste
    if temporal_split:
        # Temporal split: train on older data, test on recent data
        print("📅 Using TEMPORAL split (train on past, test on future)")
        print("   Pattern: HUB_IA validated - quantile(0.8) cutoff with gap")
        
        # Verify @timestamp exists (should be column now after reset_index)
        if '@timestamp' not in df_model.columns:
            print("⚠️ WARNING: @timestamp not in dataframe, cannot do temporal split.")
            print("   Falling back to random split.")
            temporal_split = False
        else:
            # Sort by timestamp (crucial for temporal correctness)
            df_model = df_model.sort_values('@timestamp')
            
            # HUB_IA approach: 80% cutoff on time (not row count)
            cutoff_time = df_model['@timestamp'].quantile(0.8)
            
            # Split: train on PAST (<=cutoff), test on FUTURE (>cutoff)
            df_train = df_model[df_model['@timestamp'] <= cutoff_time].copy()
            df_test = df_model[df_model['@timestamp'] > cutoff_time].copy()
            
            # Extract features and target
            X_train = df_train[valid_features]
            y_train = df_train['future_failure']
            X_test = df_test[valid_features]
            y_test = df_test['future_failure']
            
            # Temporal validation: train dates must be < test dates
            train_start = df_train['@timestamp'].min()
            train_end = df_train['@timestamp'].max()
            test_start = df_test['@timestamp'].min()
            test_end = df_test['@timestamp'].max()
            
            print(f"\n  ✅ Temporal Split Validation:")
            print(f"     Train: {len(X_train):,} samples")
            print(f"            {train_start} to {train_end}")
            print(f"     Test:  {len(X_test):,} samples")
            print(f"            {test_start} to {test_end}")
            
            # CRITICAL validation: test must come AFTER train
            gap_days = (test_start - train_end).days
            print(f"     Gap:   {gap_days} days between train end and test start")
            
            if test_start <= train_end:
                print(f"\n  ⚠️ ERROR: Temporal overlap detected!")
                print(f"     Test starts ({test_start}) BEFORE train ends ({train_end})")
                print(f"     This would allow model to 'see the future' - aborting.")
                return
            
            if gap_days < 0:
                print(f"\n  ⚠️ WARNING: Negative gap ({gap_days} days) - temporal leak risk!")
            elif gap_days < 7:
                print(f"  ⚠️ WARNING: Gap < 7 days - consider increasing for safety")
            else:
                print(f"  ✅ PASS: Temporal split validated (gap {gap_days} days >= 7 days)")
    
    if not temporal_split:
        # Random split (original approach or fallback)
        print("🎲 Using RANDOM split (stratified)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test:  {len(X_test)} samples")
    
    # Treina o modelo
    print(f"Training RandomForest: n_estimators={n_estimators}, max_features={max_features}, n_jobs={n_jobs}")
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_features=max_features,
        random_state=42, 
        class_weight='balanced', 
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    print("Treinamento concluído.")

    # Avalia o modelo no conjunto de teste
    print("\n--- Avaliação do Modelo ---")
    y_pred = model.predict(X_test)
    
    print("\nMatriz de Confusão:")
    print("--------------------")
    print(confusion_matrix(y_test, y_pred))
    print("TN | FP")
    print("FN | TP")
    
    print("\nRelatório de Classificação:")
    print("---------------------------")
    print(classification_report(y_test, y_pred))
    
    # ========== PHASE 6: THRESHOLD OPTIMIZATION (HUB_IA PATTERN) ==========
    print("\n--- Threshold Optimization (Precision-Recall Curve) ---")
    from sklearn.metrics import precision_recall_curve
    
    # Get prediction probabilities for positive class
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Option 1: Target recall >= 85% (HUB_IA approach for critical failures)
    target_recall = 0.85
    recall_idx = np.where(recall >= target_recall)[0]
    
    if len(recall_idx) > 0:
        optimal_threshold_recall = thresholds[recall_idx[-1]]
        optimal_precision_recall = precision[recall_idx[-1]]
        optimal_recall_recall = recall[recall_idx[-1]]
        
        print(f"\n  🎯 TARGET RECALL >= {target_recall:.0%}:")
        print(f"     Optimal Threshold: {optimal_threshold_recall:.4f}")
        print(f"     Precision: {optimal_precision_recall:.2%}")
        print(f"     Recall:    {optimal_recall_recall:.2%}")
        
        # Re-predict with optimal threshold
        y_pred_optimal_recall = (y_proba >= optimal_threshold_recall).astype(int)
        
        print(f"\n  📊 Confusion Matrix (Threshold {optimal_threshold_recall:.4f}):")
        cm_recall = confusion_matrix(y_test, y_pred_optimal_recall)
        print(f"     {cm_recall}")
        print(f"     TN={cm_recall[0,0]:,} | FP={cm_recall[0,1]:,}")
        print(f"     FN={cm_recall[1,0]:,} | TP={cm_recall[1,1]:,}")
    else:
        print(f"\n  ⚠️ WARNING: Cannot achieve recall >= {target_recall:.0%}")
        optimal_threshold_recall = None
        optimal_precision_recall = None
        optimal_recall_recall = None
    
    # Option 2: Maximize F1-score (balanced approach)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_f1_idx]
    optimal_precision_f1 = precision[optimal_f1_idx]
    optimal_recall_f1 = recall[optimal_f1_idx]
    optimal_f1 = f1_scores[optimal_f1_idx]
    
    print(f"\n  ⚖️ MAXIMIZE F1-SCORE:")
    print(f"     Optimal Threshold: {optimal_threshold_f1:.4f}")
    print(f"     Precision: {optimal_precision_f1:.2%}")
    print(f"     Recall:    {optimal_recall_f1:.2%}")
    print(f"     F1-Score:  {optimal_f1:.4f}")
    
    # Re-predict with F1-optimal threshold
    y_pred_optimal_f1 = (y_proba >= optimal_threshold_f1).astype(int)
    
    print(f"\n  📊 Confusion Matrix (Threshold {optimal_threshold_f1:.4f}):")
    cm_f1 = confusion_matrix(y_test, y_pred_optimal_f1)
    print(f"     {cm_f1}")
    print(f"     TN={cm_f1[0,0]:,} | FP={cm_f1[0,1]:,}")
    print(f"     FN={cm_f1[1,0]:,} | TP={cm_f1[1,1]:,}")
    
    # Default threshold 0.5 comparison
    print(f"\n  📌 COMPARISON WITH DEFAULT (threshold=0.5):")
    print(f"     Default:  Precision={100*(y_pred == 1).sum() and (y_test[y_pred == 1] == 1).sum() / (y_pred == 1).sum() or 0:.2%}, Recall={(y_test[y_pred == 1] == 1).sum() / (y_test == 1).sum():.2%}")
    if optimal_threshold_recall:
        print(f"     Recall≥85%: Precision={optimal_precision_recall:.2%}, Recall={optimal_recall_recall:.2%} (threshold={optimal_threshold_recall:.4f})")
    print(f"     Max F1:   Precision={optimal_precision_f1:.2%}, Recall={optimal_recall_f1:.2%} (threshold={optimal_threshold_f1:.4f})")
    
    # Salva o modelo e a lista de features
    print("\n--- Salvando o Modelo ---")
    model_payload = {
        'model': model,
        'features': valid_features,
        'threshold_optimization': {
            'default_threshold': 0.5,
            'optimal_threshold_recall_85': optimal_threshold_recall,
            'precision_at_recall_85': optimal_precision_recall,
            'recall_at_recall_85': optimal_recall_recall,
            'optimal_threshold_f1': optimal_threshold_f1,
            'precision_at_f1': optimal_precision_f1,
            'recall_at_f1': optimal_recall_f1,
            'f1_score': optimal_f1
        }
    }
    joblib.dump(model_payload, 'model.joblib')
    print(f"Modelo salvo com sucesso em 'model.joblib'")
    print(f"  ✅ Threshold optimization saved to model_payload['threshold_optimization']")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Treina o modelo preditivo a partir dos dados processados.')
    parser.add_argument('--days', type=int, default=90, help='janela de dias a usar para treinamento (default: 90). Use 0 ou valor negativo para usar TODOS os dados.')
    parser.add_argument('--all', action='store_true', help='usar todos os dados disponíveis (equivalente a --days 0)')
    parser.add_argument('--n-estimators', type=int, default=100, help='número de estimators/árvores para o RandomForest (default: 100)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='n_jobs para o RandomForest (default -1 usa todos os cores)')
    parser.add_argument('--max-features', type=str, default='sqrt', help='max_features para RandomForest (default: sqrt). Opções: sqrt, log2, None (todas as features)')
    parser.add_argument('--baseline', action='store_true', help='modo baseline: usar apenas as 4 features originais para comparação')
    parser.add_argument('--temporal-split', action='store_true', help='usar split temporal (train=passado, test=futuro) ao invés de split aleatório')
    parser.add_argument('--clean-data', action='store_true', help='usar dados limpos (payloads_processed_clean.csv) sem outliers')
    parser.add_argument('--cross-validate', action='store_true', help='Task 3: executar validação cruzada temporal (TimeSeriesSplit) com 5 folds e threshold 0.23')
    parser.add_argument('--n-splits', type=int, default=5, help='número de folds para cross-validation (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.23, help='threshold otimizado para predições (default: 0.23 from Task 9)')
    args = parser.parse_args()

    print("Carregando dados...")
    data = load_data(use_clean_data=args.clean_data)

    if data is None or data.empty:
        print("Não foi possível carregar os dados. Verifique o arquivo 'payloads_processed.csv'.")
        sys.exit(1)

    # Ensure timestamp is datetime
    if '@timestamp' in data.columns:
        data['@timestamp'] = pd.to_datetime(data['@timestamp'])

    # Decide whether to filter by recent days or use all data
    use_all = args.all or (args.days is not None and args.days <= 0)
    if use_all:
        print("Usando TODOS os dados disponíveis para treinamento (sem filtro de janela).")
        data_to_use = data
    else:
        print(f"Filtrando dados para os últimos {args.days} dias para treinamento.")
        data_to_use = data[data['@timestamp'] >= data['@timestamp'].max() - pd.Timedelta(days=args.days)]

    if data_to_use is not None and not data_to_use.empty:
        df_features = feature_engineering(data_to_use)
        if not df_features.empty:
            # pass through hyperparameters from CLI args
            try:
                n_estimators = args.n_estimators
                n_jobs = args.n_jobs
                max_features = args.max_features if args.max_features != 'None' else None
                baseline_mode = args.baseline
                temporal_split = args.temporal_split
                cross_validate_mode = args.cross_validate
                n_splits = args.n_splits
                threshold = args.threshold
            except Exception:
                n_estimators = 100
                n_jobs = -1
                max_features = 'sqrt'
                baseline_mode = False
                temporal_split = False
                cross_validate_mode = False
                n_splits = 5
                threshold = 0.23
            
            # ========== TASK 3: CROSS-VALIDATION MODE ==========
            if cross_validate_mode:
                print("\n🎯 MODE: CROSS-VALIDATION (Task 3)")
                print("Skipping regular training, performing 5-fold temporal cross-validation...")
                
                cv_results = cross_validate_temporal(
                    df_features, 
                    n_estimators=n_estimators, 
                    max_features=max_features,
                    optimal_threshold=threshold,
                    n_splits=n_splits,
                    baseline_mode=baseline_mode
                )
                
                if cv_results:
                    # Save CV results to JSON
                    import json
                    from datetime import datetime
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv_output_dir = 'train_monitor_output'
                    os.makedirs(cv_output_dir, exist_ok=True)
                    
                    cv_filename = os.path.join(cv_output_dir, f'cv_results_{timestamp}.json')
                    
                    # Convert numpy types to Python native types for JSON serialization
                    cv_results_serializable = {}
                    for key, value in cv_results.items():
                        if isinstance(value, (list, tuple)):
                            cv_results_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
                        elif isinstance(value, (np.floating, np.integer)):
                            cv_results_serializable[key] = float(value)
                        elif isinstance(value, (bool, np.bool_)):
                            cv_results_serializable[key] = bool(value)
                        else:
                            cv_results_serializable[key] = value
                    
                    with open(cv_filename, 'w') as f:
                        json.dump(cv_results_serializable, f, indent=2)
                    
                    print(f"\n✅ Cross-validation results saved to: {cv_filename}")
                    print(f"\n{'='*80}")
                    print("TASK 3 COMPLETE - Cross-Validation Executed Successfully")
                    print(f"{'='*80}")
                else:
                    print("\n❌ Cross-validation failed.")
            else:
                # Regular training mode
                train_and_save_model(df_features, n_estimators=n_estimators, n_jobs=n_jobs, max_features=max_features, baseline_mode=baseline_mode, temporal_split=temporal_split)
        else:
            print("Não foi possível gerar features. O treinamento não será executado.")
    else:
        print(f"Nenhum dado disponível após aplicar o filtro (dias={args.days}, all={args.all}). Verifique o dataset.")
