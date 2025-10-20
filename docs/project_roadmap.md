# Project Roadmap: Predictive Maintenance System

**Last Updated:** 2025-10-15

## Executive Summary

Comprehensive predictive maintenance system for IoT sensors using machine learning to detect battery failures and device health issues. **Current Focus:** Detecting battery defects through oscillation/instability patterns BEFORE total failure (<2.5V) for preventive maintenance.

## Development Timeline

### ✅ Phase 0: Baseline (Oct 10, 2025)
**Status:** COMPLETED  
**Goal:** Establish minimal viable predictor

- **Features:** 4 baseline features (battery rolling mean/std, RSSI rolling mean/std)
- **Model:** RandomForest, 99.32% recall, 22.8 MB
- **Architecture:** Offline training (`train_model.py`) + Streamlit dashboard (`predictive_analysis.py`)
- **Integration:** SHAP explanations for interpretability

### ✅ Phase 1: Feature Expansion (Oct 12, 2025)
**Status:** COMPLETED & DEPLOYED  
**Goal:** Expand feature space to capture communication patterns, LoRa metrics, device context

- **Features Added:** 8 new features (total 12)
  - Communication: `communication_errors`, `error_rate`
  - LoRa: `lora_snr_rolling_mean_7d`, `lora_snr_rolling_std_7d`
  - Device Context: `deployment_age`, `msg_type_variety`, `active_days_7d`, `last_rsrp_rolling_mean_7d`
- **Results:** 99.95% recall (+92.6% improvement over baseline), 2.98 MB (87% smaller)
- **Key Discovery:** deployment_age and communication_errors became top 2 features

### ✅ Phase 2: Raw Battery + Temporal Split (Oct 14, 2025)
**Status:** COMPLETED & DEPLOYED  
**Goal:** Fix rolling mean masking issue (device at 2.78V showing only 3-5% risk)

- **Problem:** Model saw rolling_mean_7d (~3.1V), not current voltage (2.78V)
- **Features Added:** `battery_voltage` (raw current value)
- **Infrastructure:** Added `--temporal-split` option, `--all` flag for full dataset
- **Training:** 435k samples, 13 features
- **Results:** 100% recall, 100% precision, 8.4 MB
- **Investigation:** Created `analysis_model_investigation.ipynb` documenting debugging process

### ✅ Phase 3: Data Cleaning + Threshold Features (Oct 14, 2025)
**Status:** COMPLETED - FAILED APPROACH  
**Goal:** Improve battery failure detection with cleaned data and critical thresholds

- **Data Quality:**
  - Created `tools/clean_data.py`
  - Removed 109k battery outliers (9.61% - values >5V, max 3642V impossible)
  - Cleaned 306k signal quality outliers (RSSI/SNR/RSRP/RSRQ)
  - Output: `payloads_processed_clean.csv` (417k valid samples)

- **Features Added:** 3 battery threshold features
  - `battery_critical`: Binary flag for <2.8V
  - `battery_very_low`: Binary flag for <2.5V
  - `battery_drop_rate_7d`: Voltage decline rate (V/day)

- **Training:** 417k samples, 16 features, 7.89 MB
- **Results:** 99.99% recall, 99.38% precision

- **CRITICAL FAILURE:**
  - New battery features ranked LAST in importance:
    - `battery_critical`: 0.13% (last place!)
    - `battery_very_low`: 0.42%
    - `battery_drop_rate_7d`: 1.00%
  - Top features still: `deployment_age` (17.5%), `communication_errors` (12.9%)

- **Root Cause:** 
  - Correlation battery_voltage vs future_failure = -0.0115 (practically zero)
  - Historical data: battery LOW is not primary failure cause
  - Threshold features don't capture PATTERN/OSCILLATION

- **User Clarification:** 
  - "Bateria é 1 dos N problemas de sensor. baixou de 2.5V morreu"
  - **Real Goal:** Detect battery defects through OSCILLATION/VARIATION as defect indicator BEFORE total failure
  - Normal battery = stable, linear decline
  - Defective battery = significant variation in 30 days
  - Use for preventive maintenance (replace unstable batteries early)

### 🔄 Phase 4: Instability Detection (Oct 15, 2025)
**Status:** IN PROGRESS  
**Goal:** Detect battery defects through oscillation patterns, not just voltage thresholds

- **Strategy Pivot:** Threshold-based → Pattern-based detection
- **Proposed Features:** 4 instability indicators
  - `battery_cv_7d`: Coefficient of variation (std/mean) - measures stability
  - `battery_spike_count_7d`: Count of abrupt changes (>0.2V delta)
  - `battery_range_7d`: Max-min in window - measures variability
  - `battery_trend_7d`: Linear regression slope - charge pattern

- **Expected Results:**
  - Pattern features should rank >5% importance (vs 0.13-1% for thresholds)
  - If still low → proceed to Phase 5 (IsolationForest)

- **Target:** 20 total features (16 existing + 4 new)

### 🎯 Phase 5: Dual Model Architecture (Conditional)
**Status:** PLANNED - Only if Phase 4 features fail  
**Goal:** Anomaly detection for battery patterns if RandomForest can't capture instability

- **Architecture:** 
  - RandomForest (existing): Predicts imminent failure probability
  - IsolationForest (new): Detects abnormal battery patterns
  
- **Implementation:**
  - Train IsolationForest on battery features only: [battery_voltage, battery_cv_7d, battery_spike_count_7d, battery_range_7d, battery_trend_7d]
  - Contamination parameter: 0.05 (5% anomaly rate)
  - Dashboard shows dual scores: "Failure Risk: X%" + "Battery Instability: Y%"

- **Validation:** Test on Enzo's 4 failed devices (861275072310101, 861275072370865, 861275072374958, 861275072344001)

## Current Production Model

- **Version:** Phase 2 (deployed Oct 14, 2025)
- **File:** `model.joblib`
- **Features:** 13 (12 from Phase 1 + battery_voltage)
- **Metrics:** 100% recall, 100% precision
- **Size:** 8.4 MB
- **Training Data:** 435k samples (full dataset with --all flag)

## Model Comparison

| Phase | Features | Size | Recall | Precision | Key Innovation |
|-------|----------|------|--------|-----------|----------------|
| Baseline | 4 | 22.8 MB | 99.32% | - | Initial proof-of-concept |
| Phase 1 | 12 | 2.98 MB | 99.95% | - | Communication + LoRa context |
| Phase 2 | 13 | 8.4 MB | 100% | 100% | Raw battery + temporal split |
| Phase 3 | 16 | 7.89 MB | 99.99% | 99.38% | Clean data (failed approach) |
| Phase 4 | 20 | TBD | TBD | TBD | Pattern-based instability |

## Research & Documentation

- **Context7 Research:** Retrieved scikit-learn best practices (IsolationForest, LocalOutlierFactor, TimeSeriesSplit)
- **Investigation Notebook:** `analysis_model_investigation.ipynb` - documents Phase 2 debugging
- **Technical Reports:**
  - `model_investigation_findings.md` - Root cause analysis
  - `data_cleaning_report_20251014_144223.md` - Outlier removal process
  - `shap_feature_importance_report.md` - Feature contribution analysis
  - `phase4_instability_detection_plan.md` - NEW (this document describes Phase 4 strategy)

## Next Steps

1. ✅ Project cleanup (archive obsolete files)
2. ✅ Update documentation
3. 🔄 Implement instability features in `train_model.py`
4. 🔄 Train Phase 4 model with pattern detection
5. ⏭️ Validate on Enzo's devices
6. ⏭️ If needed: Implement IsolationForest (Phase 5)
