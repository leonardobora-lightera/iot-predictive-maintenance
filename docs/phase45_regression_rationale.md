# 🔄 Phase 4.5 - Regression to Battery-Centric Baseline

**Data:** 17 de Outubro de 2025  
**Decisão:** Regressão estratégica de Phase 6 (27 features) para Phase 4.5 (8-9 features battery-only)  
**Razão:** Instabilidade temporal severa (recall 12-85% entre folds) causada por feature complexity  
**Status:** ✅ DECISÃO TOMADA - Implementação pendente  

---

## 🎯 Contexto da Decisão

### Problema Identificado (Task 3 - Cross-Validation)

**Phase 6 Model (27 features + threshold 0.23):**
- ✅ 80/20 temporal split: 85.06% recall (Task 9 - MISLEADING)
- ❌ 5-fold cross-validation: 49.6% mean recall (Task 3 - REALITY CHECK)
- ❌ Instabilidade SEVERA: fold 1 (12%), fold 2 (43%), fold 3 (22%), fold 4 (84%), fold 5 (85%)
- ❌ CV = 61.4% (target era <10%)
- ❌ is_stable = false

**Root Cause Hypothesis:**
1. **Feature Dilution:** Signal/comm features não correlacionadas com battery failure adicionam ruído
2. **Temporal Overfitting:** Lag_diff features funcionam em períodos específicos (folds 4-5) mas falham em outros (folds 1-3)
3. **Lack of Focus:** POC começou focado (battery instability) mas expandiu demais (IoT genérico)

---

## 🔍 Comparação: STI Dashboard vs Nossa Implementação

### HUB_IA (STI Dashboard - Fibra Óptica)
- **Domínio:** Específico - Sensores de fibra óptica (Lightera/Fibrasil)
- **Target:** Falhas acionáveis em `optical_power` (queda de sinal)
- **Features:** 37 (9 raw + 28 engineered) - TODAS relacionadas ao domínio
- **Foco:** ✅ CLARO - prever falha de SINAL de fibra óptica
- **Resultado:** ✅ Deploy em produção com Streamlit app

### Nossa Implementação (Tentativa IoT Genérico)
- **Domínio:** Amplo - Sensores IoT genéricos (LoRa/LTE)
- **Target:** Falhas de BATERIA (`battery_voltage < 2.5V` em 30 dias)
- **Features:** 27 (mix battery + signal + communication + lag_diff)
  - 10 battery features (CORE)
  - 4 signal features (rssi, snr, rsrp, rsrq) - ❌ NÃO correlacionadas com battery
  - 3 communication features (f_cnt, total_errors) - ❌ INDIRETAS
  - 5 lag_diff features (Task 6) - ⚠️ OVERFITTING temporal
  - 2 context features (deployment_age, msg_type_error_pct)
- **Foco:** ❌ DILUÍDO - tentando ser genérico perdeu foco no core problem
- **Resultado:** ❌ Instabilidade temporal severa (CV 61.4%)

**Lesson Learned:** HUB_IA tinha escopo específico e funcionou. Nós precisamos FOCAR em battery health prediction, não IoT genérico.

---

## 🎯 Phase 4.5 - Battery-Centric Baseline

### Objetivo
**Provar que battery failure prediction é viável e estável** antes de adicionar complexidade.

### Features Selecionadas (8-9 features FOCADAS)

#### Core Battery Health (3 features)
1. **`battery_voltage`** (raw) - Valor atual - **CRÍTICO**
2. **`battery_rolling_mean_7d`** - Tendência média (7 dias)
3. **`battery_rolling_std_7d`** - Variabilidade

#### Instability Detection (3 features) - **NOSSA INOVAÇÃO**
4. **`battery_range_7d`** - Amplitude (max-min) - detecta oscilação
5. **`battery_cv_7d`** - Coeficiente de variação (std/mean) - instabilidade relativa
6. **`battery_spike_count_7d`** - Contagem de mudanças >0.2V - spikes abruptos

#### Threshold Detection (2 features)
7. **`battery_critical`** - Binário (<2.8V) - threshold warning
8. **`battery_very_low`** - Binário (<2.5V) - threshold crítico

#### Context (1 feature - OPCIONAL)
9. **`deployment_age_days`** - Idade do device (correlação com degradação)

**Total:** 8-9 features **EXCLUSIVAMENTE de bateria**

### Features REMOVIDAS (Rationale)

#### Signal Features (4 features) - ❌ REMOVIDAS
- `rssi_rolling_mean_7d`, `rssi_rolling_std_7d`
- `snr_rolling_mean_7d`, `snr_rolling_std_7d`
- **Razão:** Não correlacionadas com battery failure. Dependem de rede/interferência, não de battery health.

#### Communication Features (3 features) - ❌ REMOVIDAS
- `f_cnt_rolling_mean_7d`, `f_cnt_rolling_std_7d`
- `total_errors_rolling_sum_7d`
- **Razão:** Indiretas - fraca correlação com battery health. Dependem de comportamento de rede.

#### Lag_diff Features (5 features) - ❌ REMOVIDAS
- `battery_lag_diff_1d`, `battery_lag_diff_7d`
- `rssi_lag_diff_7d`, `snr_lag_diff_7d`
- `f_cnt_lag_diff_7d`
- **Razão:** Overfitting temporal - funcionam em períodos específicos mas não generalizam. Causaram instabilidade (fold 1-3: 12-43% vs fold 4-5: 84-85%).

#### Message Type Error (1 feature) - ❌ REMOVIDA
- `msg_type_error_pct_7d`
- **Razão:** Indireta - não core para battery prediction.

---

## 🔬 Hipóteses a Validar

### Hipótese 1: Feature Complexity Causa Instabilidade
- **Teste:** Phase 4.5 (8 features battery-only) deve ter CV < 20% (vs Phase 6 CV 61.4%)
- **Se TRUE:** Provado que signal/comm/lag_diff adicionaram ruído
- **Se FALSE:** Problema é mais profundo (concept drift real, target definition)

### Hipótese 2: Battery-Only Features São Suficientes
- **Teste:** Phase 4.5 deve atingir recall ≥70% consistente (vs Phase 6 mean 49.6%)
- **Se TRUE:** Core problem (battery prediction) é solucionável com foco
- **Se FALSE:** Precisamos repensar target definition ou data quality

### Hipótese 3: Instability Detection Tem Valor
- **Teste:** `battery_range_7d` e `battery_cv_7d` devem estar no top 5 feature importance
- **Se TRUE:** Nossa inovação (instability detection) é válida
- **Se FALSE:** Threshold detection (battery_critical, battery_very_low) é suficiente

---

## ⏱️ Plano de Implementação (5 horas)

### PASSO 1: Simplificar Features (1h)
```python
# Criar train_model_phase45.py (battery-only)
valid_features = [
    'battery_voltage',
    'battery_rolling_mean_7d',
    'battery_rolling_std_7d',
    'battery_range_7d',
    'battery_cv_7d',
    'battery_spike_count_7d',
    'battery_critical',
    'battery_very_low',
    'deployment_age_days'  # opcional
]
```

### PASSO 2: Fix Temporal Split (30min)
```python
# Preservar @timestamp ANTES de dropna()
df_model = df_model.reset_index()  # @timestamp vira coluna
df_model = df_model.dropna(subset=valid_features)

# Temporal split 80/20
cutoff_time = df_model['@timestamp'].quantile(0.8)
df_train = df_model[df_model['@timestamp'] <= cutoff_time]
df_test = df_model[df_model['@timestamp'] > cutoff_time]

# Validação
print(f"Train: {df_train['@timestamp'].min()} to {df_train['@timestamp'].max()}")
print(f"Test:  {df_test['@timestamp'].min()} to {df_test['@timestamp'].max()}")
# CRITICAL: test.min() > train.max() for truly temporal split
```

### PASSO 3: Train + Threshold Optimization (1h)
```python
# Train RandomForest
model = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # sqrt(8) ≈ 3 features por split
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Precision-Recall curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Target recall >= 85% (ou ajustar baseado em resultados)
target_recall = 0.85
recall_idx = np.where(recall >= target_recall)[0]
optimal_threshold = thresholds[recall_idx[-1]]
```

### PASSO 4: Cross-Validation (2h)
```python
# TimeSeriesSplit 5-fold, gap 7 dias
from sklearn.model_selection import TimeSeriesSplit

# Calculate gap in samples
avg_samples_per_day = len(df_model) / (df_model['@timestamp'].max() - df_model['@timestamp'].min()).days
gap_samples = int(7 * avg_samples_per_day)

tscv = TimeSeriesSplit(n_splits=5, gap=gap_samples)

fold_results = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Train
    model_fold = RandomForestClassifier(**params).fit(X[train_idx], y[train_idx])
    
    # Predict with optimal threshold
    y_proba_fold = model_fold.predict_proba(X[test_idx])[:, 1]
    y_pred_fold = (y_proba_fold >= optimal_threshold).astype(int)
    
    # Metrics
    fold_recall = recall_score(y[test_idx], y_pred_fold)
    fold_precision = precision_score(y[test_idx], y_pred_fold)
    fold_results.append({'fold': fold+1, 'recall': fold_recall, 'precision': fold_precision})

# Stability
mean_recall = np.mean([r['recall'] for r in fold_results])
std_recall = np.std([r['recall'] for r in fold_results])
cv_recall = std_recall / mean_recall if mean_recall > 0 else float('inf')

print(f"Mean Recall: {mean_recall:.4f}")
print(f"Std Recall: {std_recall:.4f}")
print(f"CV: {cv_recall:.4f} (target < 0.20)")
print(f"Stable: {cv_recall < 0.20}")
```

**Target de Estabilidade:** CV(recall) < 20% (vs Phase 6 CV 61.4%)

### PASSO 5: Documentar e Decidir (30min)
- Criar `phase45_battery_centric_results.md`
- Comparar Phase 4.5 (8 features) vs Phase 6 (27 features)
- **Decisão:**
  - Se CV < 20% e recall ≥70% → ✅ SUCESSO → incremental features opcionais
  - Se CV ≥ 20% ou recall <70% → ❌ Investigar concept drift ou target definition

---

## 📊 Critérios de Sucesso

### Mínimo Viável (MVP)
- ✅ CV(recall) < 30% (vs 61.4% atual) - **MELHORIA 2x**
- ✅ Mean recall ≥ 60% (vs 49.6% atual) - **MELHORIA 20%**
- ✅ Recall em CADA fold ≥ 40% (vs fold 1-3: 12-43%) - **SEM CATASTROPHIC FAILURES**

### Target Ideal
- 🎯 CV(recall) < 20% - **ESTÁVEL para produção**
- 🎯 Mean recall ≥ 70% - **ACEITÁVEL para POC**
- 🎯 Recall em CADA fold ≥ 60% - **CONSISTENTE entre períodos**

### Validação de Inovação
- 🎯 `battery_range_7d` e `battery_cv_7d` no top 5 feature importance - **INSTABILITY DETECTION WORKS**

---

## 🎓 Lessons Learned

### ✅ Do's
1. **Focus on Core Problem:** Battery failure prediction, não IoT genérico
2. **Domain-Specific Features:** Apenas features diretamente relacionadas ao target
3. **Incremental Complexity:** Prove baseline funciona ANTES de expandir
4. **Cross-Validation First:** Single split pode enganar, CV revela verdade

### ❌ Don'ts
1. **Feature Dilution:** Adicionar features "porque sim" sem correlação com target
2. **Temporal Overfitting:** Lag_diff pode overfit em padrões temporais específicos
3. **Blind Expansion:** Expandir features sem validar estabilidade
4. **Misleading Metrics:** 85% recall em single split ≠ 49.6% em CV

---

## 🔄 Próximos Passos (Após Phase 4.5)

### Se Estável (CV < 20%)
1. Adicionar features incrementalmente (signal, comm) SE houver correlação com battery
2. Re-testar lag_diff com regularização mais forte (max_depth, min_samples_leaf)
3. Explorar ensemble temporal (treinar modelo por período e combinar)

### Se Ainda Instável (CV ≥ 20%)
1. Investigar concept drift temporal (distribuição de falhas por período)
2. Revisar target definition (2.5V em 30 dias é uniforme ao longo do tempo?)
3. Analisar data quality (devices diferentes têm padrões diferentes?)
4. Considerar re-balanceamento temporal (SMOTE/undersampling por fold)

---

## 📂 Arquivos Movidos

**Diretório:** `archive/phase6_failed_experiments/`

### Documentação
- `docs/task9_threshold_optimization_results.md` - Misleading 85% recall (single split)
- `docs/task6_lag_diff_implementation.md` - Lag_diff features (overfitting temporal)
- `docs/task6_validation_results.md` - Validation que falhou em generalizar

### Resultados
- `cv_results/cv_results_20251017_135149.json` - Task 3 CV (instabilidade 12-85%)
- `cv_results/cv_results_20251017_141535.json` - Task 3 CV final (mean 49.6%)

### Modelos
- `models/model_phase6_27features.joblib` - Phase 6 model (27 features instável)

---

## 🎯 Conclusão

**Decisão:** Regressão para Phase 4.5 (Battery-Centric) é a estratégia correta.

**Rationale:**
1. ✅ **Focus:** Alinha com core problem (battery failure), não IoT genérico
2. ✅ **Efficiency:** 5h implementação vs semanas investigando 27 features
3. ✅ **Validation:** Testa hipótese de feature complexity de forma controlada
4. ✅ **Pragmatic:** Alinhado com espírito de POC (prove viability, then scale)

**Expected Outcome:**
- Best case: CV < 20%, recall ≥70% → Battery prediction é viável, expande incrementalmente
- Worst case: CV ≥ 20% → Problema mais profundo (concept drift, target definition), pivota estratégia

**Confidence Level:** 🔸 **MÉDIO-ALTO** - Feature dilution é hipótese forte, mas não garantida.

---

**Last Updated:** 2025-10-17  
**Next Action:** Implementar Phase 4.5 (train_model_phase45.py)  
**Duration Estimate:** 5 horas para resposta definitiva  
**Decision Point:** CV < 20% = SUCCESS, CV ≥ 20% = INVESTIGATE DEEPER
