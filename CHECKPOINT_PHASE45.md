# 📌 Phase 4.5 Battery-Centric Baseline - Checkpoint

**Data:** 17 de Outubro de 2025  
**Status:** ✅ CHECKPOINT CRIADO - Pronto para implementação  
**Baseline:** Phase 4.5 (8-9 features battery-only)  

---

## 🎯 Estado Atual do Projeto

### Decisão Estratégica
**REGRESSÃO para Battery-Centric Baseline** devido à instabilidade temporal da Phase 6.

### Experimentos Arquivados
- Phase 6 (27 features): Recall 49.6% mean (CV 61.4% - INSTÁVEL)
- Task 9 (threshold optimization): 85% recall misleading (apenas em specific split)
- Task 3 (cross-validation): Instabilidade 12-85% entre folds

---

## 🧬 Features da Baseline Phase 4.5

### Core Battery Health (3 features)
1. `battery_voltage` (raw)
2. `battery_rolling_mean_7d`
3. `battery_rolling_std_7d`

### Instability Detection (3 features) - INOVAÇÃO
4. `battery_range_7d`
5. `battery_cv_7d`
6. `battery_spike_count_7d`

### Threshold Detection (2 features)
7. `battery_critical` (<2.8V)
8. `battery_very_low` (<2.5V)

### Context (1 feature - opcional)
9. `deployment_age_days`

**Total:** 8-9 features FOCADAS em battery health

---

## 🎯 Critérios de Sucesso

### Mínimo Viável
- CV(recall) < 30% (vs 61.4% Phase 6)
- Mean recall ≥ 60% (vs 49.6% Phase 6)
- Recall em CADA fold ≥ 40%

### Target Ideal
- CV(recall) < 20% (ESTÁVEL)
- Mean recall ≥ 70% (ACEITÁVEL)
- Recall em CADA fold ≥ 60% (CONSISTENTE)

---

## 📂 Arquivos Essenciais Preservados

### Data
- `data/processed/payloads_processed_clean.csv` (417k samples)
- `tools/clean_data.py` (outlier removal)

### Core ML
- `src/collector/train_model.py` (a ser simplificado para Phase 4.5)
- `tools/train_monitor.py` (resource monitoring)

### Documentation
- `docs/ml_implementation_status.md` (atualizado para Phase 4.5)
- `docs/phase45_regression_rationale.md` (decisão documentada)
- `docs/temporal_split_implementation_plan.md` (fix técnico)
- `docs/historical_comparison_analysis.md` (HUB_IA learnings)

---

## 🔄 Próximos Passos

1. **Implementar Phase 4.5** (5h)
   - Simplificar train_model.py para 8-9 features battery-only
   - Fix temporal split (reset_index antes dropna)
   - Train + threshold optimization
   - Cross-validation 5-fold

2. **Validar Estabilidade**
   - Se CV < 20% → SUCESSO → incremental features
   - Se CV ≥ 20% → Investigar concept drift

3. **Documentar Resultados**
   - Criar phase45_battery_centric_results.md
   - Comparar Phase 4.5 vs Phase 6
   - Decisão sobre próximos passos

---

**Checkpoint:** 2025-10-17  
**Commit:** Pendente (após criar este checkpoint)  
**Next Action:** Implementar train_model_phase45.py
