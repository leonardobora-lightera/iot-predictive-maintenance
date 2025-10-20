# 📊 Análise Comparativa: Nossa Implementação vs HUB_IA (Sprint 3/4)

**Data:** 17 de Outubro de 2025  
**Contexto:** Engenharia de Contexto - Aprendizado de Projetos Históricos  
**Fontes:** HUB_IA Sprint 3 e 4 (Lightera/Fibrasil - Lucas do Rio Verde)

---

## 🎯 Executive Summary

**HUB_IA Sprints 3/4** representa implementação completa de ML preditivo para sensores de fibra óptica, com:
- ✅ **Temporal split correto** (80/20)
- ✅ **Deploy em produção** (Streamlit app)
- ✅ **Modelo com assertividade aceitável** (RandomForest + XGBoost)
- ✅ **37 features engineered** (rolling windows 6h + lag_diff)

**Nossa Implementação** (POC IoT Genérico):
- ✅ **22 features** (instability detection - inovação)
- ⚠️ **Temporal split BUGADO** (CRÍTICO a corrigir)
- ✅ **99.93% recall** (possivelmente otimista por random split)
- ✅ **Dashboard profissional** (4 páginas + SHAP)

---

## 📦 Contexto Histórico

### Projetos Anteriores
1. **Confluence:** Pipeline, arquitetura, notebooks (não especificado sprint)
2. **IFCE:** Notebook experimental, **NÃO deployado**, parou no notebook, sem entrega definida ❌
3. **SENAI:** Usou IFCE como base, 1 mês e meio de desenvolvimento
4. **HUB_IA Sprint 3:** Modelagem inicial (RandomForest + XGBoost)
5. **HUB_IA Sprint 4:** Deploy completo com Streamlit app ✅

### Cliente & Contexto
- **Cliente:** Lightera (sensores fibra óptica)
- **Local:** Lucas do Rio Verde/Fibrasil
- **Problema:** Detecção de falhas acionáveis em fibra óptica
- **Domínio:** Optical power, accelerometer (movimento de cabos), temperature, battery

### Relevância para EyOn
- **RN IMPORTANTE:** IA no EyOn é crucial
- **Foco:** Problem Discovery (dashboard facilita descoberta de padrões)
- **Nossa abordagem:** 4-page structure + SHAP explicabilidade alinhada

---

## 🔬 Análise Técnica Comparativa

### 1. Features Engineering

#### HUB_IA (37 features)
**Raw Features (9):**
```python
# Sensores físicos
accelerometer_pitch, accelerometer_roll, accelerometer_yaw
battery, light_lux, temperature, rssi, counter

# Domínio específico: Fibra Óptica
optical_power_level, optical_power_1490nm
```

**Engineered Features (28):**
```python
# Temporais (3)
hour, day_of_week, month

# Magnitude de acelerômetro (derivada)
accelerometer_magnitude

# Rolling Windows 6h - Mean/Std (14 features)
accelerometer_magnitude_roll_mean_6h, _roll_std_6h
temperature_roll_mean_6h, _roll_std_6h
battery_roll_mean_6h, _roll_std_6h
light_lux_roll_mean_6h, _roll_std_6h
optical_power_1490nm_roll_mean_6h, _roll_std_6h
rssi_roll_mean_6h, _roll_std_6h
counter_roll_mean_6h, _roll_std_6h
optical_power_level_roll_mean_6h, _roll_std_6h

# Lag Differences 6h (7 features) ⭐ INOVAÇÃO!
accelerometer_magnitude_lag_diff_6h
temperature_lag_diff_6h
battery_lag_diff_6h
light_lux_lag_diff_6h
optical_power_1490nm_lag_diff_6h
rssi_lag_diff_6h
counter_lag_diff_6h
optical_power_level_lag_diff_6h  # ⭐ FEATURE MAIS IMPORTANTE!
```

**Janela Temporal:** 6 horas (mais granular que nossa)

---

#### Nossa Implementação (22 features)

**Battery Features (10):**
```python
battery_voltage  # Raw (Phase 2)
battery_rolling_mean_7d, battery_rolling_std_7d
battery_critical, battery_very_low  # Thresholds
battery_drop_rate_7d  # V/day decline
battery_cv_7d, battery_range_7d  # ⭐ INSTABILITY (Phase 4)
battery_spike_count_7d, battery_trend_7d  # ⭐ INSTABILITY (Phase 4)
```

**Signal Quality (4):**
```python
rssi_rolling_mean_7d, rssi_rolling_std_7d
snr_rolling_mean_7d, snr_rolling_std_7d
```

**Communication (3):**
```python
f_cnt_rolling_mean_7d, f_cnt_rolling_std_7d
total_errors_rolling_sum_7d
```

**Context (2):**
```python
deployment_age_days  # ⭐ TOP FEATURE (17.5% importance!)
msg_type_error_pct_7d
```

**Ausentes (3):** lost_payloads, registration_time, rsrp/rsrq (colunas não existem no dataset)

**Janela Temporal:** 7 dias (menos granular, mas mais contexto)

---

### 🔍 Gap Analysis: Features

| Categoria | HUB_IA | Nossa Impl | Gap |
|-----------|--------|-----------|-----|
| **Raw sensors** | 9 | 1 (battery_voltage) | ⚠️ Faltam raw features |
| **Rolling windows** | 14 (6h) | 12 (7d) | ✅ Comparable |
| **Lag differences** | 7 (6h) | 0 | ❌ **CRÍTICO!** |
| **Temporal** | 3 (hour/day/month) | 0 | ⚠️ Missing |
| **Instability** | 1 (accel magnitude) | 6 (battery CV/range/spikes/trend) | ✅ **NOSSA INOVAÇÃO!** |
| **Thresholds** | 0 | 2 (critical/very_low) | ✅ Domain knowledge |
| **Context** | 0 | 2 (deployment_age/msg_errors) | ✅ **NOSSA INOVAÇÃO!** |

**💡 Insights:**
1. ✅ **Nossa instability detection** (battery CV/range/spikes/trend) é **única** e validada (8.01% importance)
2. ❌ **Lag_diff features** (mudanças abruptas) ausentes - HUB_IA provou valor (optical_power_lag_diff_6h foi TOP feature)
3. ✅ **deployment_age_days** é **nossa descoberta** (17.5% importance - TOP 1!)
4. ⚠️ **Raw features limitadas** - só temos battery_voltage, eles têm 9 sensors raw

---

### 2. Target Variable

#### HUB_IA
```python
# Target: 'is_failure_soon_filtered'
# Combinação de DOIS critérios (regra de negócio):

# 1. Falha detectada (is_failure_soon == 1)
# 2. Mudança óptica significativa (abs(optical_power_lag_diff_6h) > 0.05)

df['is_failure_soon_filtered'] = (
    (df['is_failure_soon'] == 1) & 
    (np.abs(df['optical_power_level_lag_diff_6h']) > 0.05)
).astype(int)
```

**Justificativa:** "Falha acionável" = falha + indicador de fibra óptica (mudança de potência)

**Recall Target:** 90% (threshold otimizado para garantir recall mínimo)

---

#### Nossa Implementação
```python
# Target: 'future_failure'
# Simples: battery_voltage < 2.5V nos próximos 30 dias

df['future_failure'] = df.groupby('device_id')['battery_voltage'].transform(
    lambda x: x.rolling('30D').min() < 2.5
).shift(-30).astype(int)
```

**Justificativa:** 2.5V = morte certa da bateria (threshold validado)

**Recall Alcançado:** 99.93% (sem otimização de threshold)

---

### 🔍 Gap Analysis: Target

| Aspecto | HUB_IA | Nossa Impl | Avaliação |
|---------|--------|-----------|-----------|
| **Complexidade** | Alta (2 critérios) | Simples (1 threshold) | ✅ Adequado ao domínio |
| **Regra de negócio** | Fibra óptica específica | Bateria genérica | ✅ Domínio diferente |
| **Janela futura** | Não especificado | 30 dias | ✅ Explícito |
| **Threshold optimization** | Sim (F1/Recall curve) | Não | ⚠️ **OPORTUNIDADE!** |

**💡 Insight:** Nossa target é adequada ao domínio IoT genérico, mas podemos **otimizar threshold** seguindo abordagem HUB_IA (maximizar recall vs precision trade-off)

---

### 3. Train/Test Split

#### HUB_IA ✅ CORRETO
```python
# Temporal split 80/20
cutoff_time = df['time'].quantile(0.8)

df_train = df[df['time'] <= cutoff_time].copy()  # Dados PASSADOS
df_test = df[df['time'] > cutoff_time].copy()    # Dados FUTUROS

print(f"Treino: {df_train['time'].min()} a {df_train['time'].max()}")
print(f"Teste: {df_test['time'].min()} a {df_test['time'].max()}")
```

**Validação:** Test dates sempre > Train dates (temporal correto)

---

#### Nossa Implementação ❌ BUGADO
```python
# TENTATIVA de temporal split (flag --temporal-split)
# MAS: @timestamp dropado durante dropna() → fallback para random split

if temporal_split:
    df_model = df_model.sort_values('@timestamp')
    split_idx = int(len(df_model) * 0.8)
    
    X_train = df_model.iloc[:split_idx][valid_features]
    X_test = df_model.iloc[split_idx:][valid_features]
    
# PROBLEMA: @timestamp não está em df_model.columns!
# Root cause: df.set_index('@timestamp') → dropna() perde index
```

**Consequência:** Random split permite modelo "ver futuro" → métricas 99.93% possivelmente otimistas em 5-10%

---

### 🔍 Gap Analysis: Split

| Aspecto | HUB_IA | Nossa Impl | Status |
|---------|--------|-----------|--------|
| **Temporal split** | ✅ Implementado | ❌ Bugado | **CRÍTICO!** |
| **Validação temporal** | ✅ Prints de datas | ⚠️ Warning silencioso | Precisa fix |
| **Gap entre train/test** | Não especificado | 7 dias planejado | ✅ Melhor que HUB_IA |
| **Cross-validation** | Não mencionado | Planejado (5-fold) | ✅ Melhor que HUB_IA |

**💡 Insight:** HUB_IA provou que **temporal split é viável e necessário**. Nossa implementação já está planejada (docs/temporal_split_implementation_plan.md), só precisa execução.

---

### 4. Modelo e Performance

#### HUB_IA
**Modelos Testados:**
1. **XGBoost** (mencionado no README, código não visível)
2. **RandomForest** (implementado no notebook, com GridSearchCV)

**Hyperparameters:**
```python
# GridSearchCV mencionado, mas parâmetros não especificados no trecho visível
best_rf_model = ...  # Modelo otimizado
```

**Threshold Optimization:**
```python
# Curva Precision-Recall para encontrar threshold ótimo
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_pred_proba_rf)

# Threshold para recall >= 90%
desired_recall_rf = 0.90
chosen_threshold_rf = ...  # Otimizado

# Recalcular com threshold escolhido
y_pred_new = (y_pred_proba_rf >= chosen_threshold_rf).astype(int)
```

**Performance:** Não especificada nos trechos lidos (precisaria executar notebook)

---

#### Nossa Implementação
**Modelo:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # sqrt(22) ≈ 4-5 features por split
    n_jobs=-1,
    class_weight='balanced',  # Compensa 11.5:1 imbalance
    random_state=42
)
```

**Performance (Random Split - possivelmente otimista):**
```
Recall:     99.93%  (6,748 TP / 6,805 actual failures)
Precision:  100%    (6,748 TP / 6,748 predictions)
F1-Score:   99.96%
FN:         57      (missed 0.07% of failures)
FP:         0       (zero false alarms!) ⭐
```

**Feature Importance Top 5:**
1. deployment_age_days: 17.5% ⭐ NOSSA DESCOBERTA
2. total_errors_rolling_sum_7d: 12.9%
3. battery_rolling_mean_7d: 10.2%
4. battery_range_7d: 8.01% ⭐ INSTABILITY
5. f_cnt_rolling_mean_7d: 6.8%

---

### 🔍 Gap Analysis: Modelo

| Aspecto | HUB_IA | Nossa Impl | Avaliação |
|---------|--------|-----------|-----------|
| **Algoritmo** | RF + XGBoost | RF only | ⚠️ Considerar XGBoost |
| **Hyperparameter tuning** | GridSearchCV | Manual (n_estimators=100) | ⚠️ **OPORTUNIDADE!** |
| **Threshold optimization** | Sim (recall curve) | Não | ⚠️ **OPORTUNIDADE!** |
| **Performance** | "Assertividade aceitável" | 99.93% recall (otimista?) | ⚠️ Precisa validação temporal |
| **Feature importance** | optical_power_lag_diff TOP | deployment_age TOP | ✅ Domínios diferentes |

**💡 Insights:**
1. ⚠️ **GridSearchCV** - HUB_IA usou, nós não. Potencial +2-3% performance.
2. ⚠️ **Threshold optimization** - HUB_IA garantiu recall 90%, nós temos 99.93% sem otimização (pode ser overfitting de random split).
3. ✅ **Nossa feature importance** revela insights únicos (deployment_age, instability) não presentes em HUB_IA.

---

### 5. Deploy e Dashboard

#### HUB_IA (Sprint 4)
**Tecnologias:**
```python
# app.py - Streamlit application
import streamlit as st
import joblib  # Carrega .pkl models
import matplotlib.pyplot as plt

# Paleta de cores Lightera (branding cliente)
PALETTE = {
    "background": "#F2F1F0",
    "text": "#262625",
    "primary": "#9D11D9",  # Roxo Lightera
    "secondary_bg": "#FFFFFF",
    "red_alert": "#FF5B5B"
}

# CSS customizado para branding
st.set_page_config(
    page_title="Painel Preditivo Lightera",
    page_icon="imagens/lightera-logo.svg",
    layout="wide"
)
```

**Estrutura:**
- Logo Lightera integrado
- Inferência com modelos .pkl pré-treinados
- Visualizações Matplotlib customizadas
- Confusion matrix, ROC curve, Precision-Recall curve
- Interface wide layout

**Armazenamento:**
- `modelos/` - Arquivos .pkl dos modelos treinados
- `dados_estruturados/model_columns.json` - Features esperadas
- `imagens/` - Assets visuais

---

#### Nossa Implementação
**Tecnologias:**
```python
# src/analysis/predictive_analysis.py
import streamlit as st
import joblib  # Carrega model.joblib
import plotly.graph_objects as go  # Gráficos interativos
import plotly.express as px
import shap  # Explicabilidade

# 4-page structure
page = st.sidebar.radio("Navegação", [
    "📊 Overview",
    "🎯 Análise de Dispositivo",
    "🧬 Jornada do Projeto",
    "📈 Insights & Top Riscos"
])
```

**Estrutura:**
- 4 páginas navegáveis
- Plotly (interativo) vs Matplotlib (estático do HUB_IA)
- SHAP explicabilidade (HUB_IA não tem)
- Gauge de risco individual
- Histórico de 30 dias (Battery/Signal/Comm tabs)
- Project Journey (Phases 0-4)

**Vantagens sobre HUB_IA:**
- ✅ **SHAP explicabilidade** (eles não têm)
- ✅ **Gráficos interativos** (Plotly > Matplotlib estático)
- ✅ **Multi-page structure** (eles têm single page)
- ✅ **Historical analysis** (30-day charts)

**Desvantagens:**
- ⚠️ **Sem branding cliente** (genérico vs Lightera customizado)
- ⚠️ **Model.joblib único** (eles têm modelos/ com múltiplos .pkl)

---

### 🔍 Gap Analysis: Deploy

| Aspecto | HUB_IA | Nossa Impl | Avaliação |
|---------|--------|-----------|-----------|
| **Deploy status** | ✅ Produção (app.py) | ✅ POC (predictive_analysis.py) | Ambos funcionais |
| **Branding** | ✅ Lightera customizado | ❌ Genérico | Depende do cliente |
| **Visualizações** | Matplotlib estático | Plotly interativo | ✅ **MELHOR!** |
| **Explicabilidade** | ❌ Sem SHAP | ✅ SHAP integrado | ✅ **MELHOR!** |
| **Estrutura** | Single page | 4 pages navegáveis | ✅ **MELHOR!** |
| **Model storage** | modelos/*.pkl múltiplos | model.joblib único | ⚠️ Considerar versionamento |

**💡 Insight:** Nossa dashboard é **tecnicamente superior** (Plotly, SHAP, multi-page), mas HUB_IA tem **branding profissional**. Para EyOn, podemos mesclar: Plotly + SHAP + branding customizado.

---

## 🎯 Lessons Learned - Ações Recomendadas

### 🚨 CRÍTICO - Temporal Split
**Lição HUB_IA:** Implementaram temporal split corretamente com validação de datas.

**Nossa Situação:** Bugado, fallback para random split → métricas otimistas.

**Ação:**
1. ✅ **Plano já existe:** `docs/temporal_split_implementation_plan.md` (38KB, 4 fases)
2. 🎯 **Executar FASE 1:** Fix básico (1-2h) - preservar @timestamp com reset_index()
3. 🎯 **Executar FASE 2:** Cross-validation (2-3h) - TimeSeriesSplit 5-fold
4. 📊 **Executar FASE 3:** Análise comparativa (3-4h) - quantificar otimismo

**Prioridade:** 🔴 MÁXIMA (todo resto depende disso)

---

### ⭐ ALTA - Lag Difference Features
**Lição HUB_IA:** `optical_power_level_lag_diff_6h` foi feature MAIS IMPORTANTE na regra de negócio (mudança >0.05).

**Nossa Situação:** Zero lag_diff features.

**Ação:**
```python
# Adicionar em feature_engineering():

# Battery lag differences (captura mudanças abruptas)
df['battery_lag_diff_1d'] = df.groupby('device_id')['battery_voltage'].diff(periods=1)  # Dia-a-dia
df['battery_lag_diff_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').apply(
    lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
).reset_index(0, drop=True)

# Signal lag differences
df['rssi_lag_diff_7d'] = ...
df['snr_lag_diff_7d'] = ...

# Communication lag differences
df['total_errors_lag_diff_7d'] = ...
```

**Justificativa:** Mudanças abruptas (spikes) são indicadores fortes de falha iminente. Rolling windows capturam tendência, lag_diff captura eventos súbitos.

**Prioridade:** 🟡 ALTA (após temporal split)

---

### ⭐ MÉDIA - Threshold Optimization
**Lição HUB_IA:** Otimizaram threshold para recall >= 90% usando curva Precision-Recall.

**Nossa Situação:** 99.93% recall com threshold padrão 0.5 (sem otimização).

**Ação:**
```python
# Adicionar em train_model.py após treinar:

from sklearn.metrics import precision_recall_curve

y_pred_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Encontrar threshold para recall >= 95% (menos agressivo que 90%)
desired_recall = 0.95
idx_recall = np.where(recall >= desired_recall)[0]
optimal_threshold = thresholds[idx_recall[-1]]

print(f"Threshold otimizado para recall >= {desired_recall}: {optimal_threshold:.4f}")

# Salvar no model payload
model_payload['optimal_threshold'] = optimal_threshold
```

**Justificativa:** Recall 99.93% pode ser "overfitting" de random split. Threshold optimization garante recall target realista após temporal split.

**Prioridade:** 🟢 MÉDIA (após temporal split + validação)

---

### ⚙️ BAIXA - GridSearchCV Hyperparameters
**Lição HUB_IA:** Usaram GridSearchCV para otimizar RandomForest.

**Nossa Situação:** Hyperparameters manuais (n_estimators=100, max_features='sqrt').

**Ação:**
```python
# Já planejado em temporal_split_implementation_plan.md
# Item #4 - Hyperparameter Tuning com Grid Search Temporal

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', 0.3],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', {0:1, 1:15}]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),  # Validação temporal!
    scoring='recall',
    n_jobs=-1
)
```

**Justificativa:** Potencial +1-3% recall vs configuração manual.

**Prioridade:** 🔵 BAIXA (após temporal split + cross-validation)

---

### 🎨 BAIXA - Branding Customizado
**Lição HUB_IA:** Paleta Lightera, logo integrado, CSS customizado.

**Nossa Situação:** Dashboard genérico.

**Ação:**
- Criar `config/theme.py` com paleta EyOn
- Adicionar logo EyOn no sidebar
- CSS customizado para cores corporativas

**Justificativa:** Profissionalismo para apresentação a stakeholders.

**Prioridade:** 🔵 BAIXA (cosmético, após funcionalidades críticas)

---

## 📚 Documentos a Consolidar/Remover

### ✅ MANTER (5 docs essenciais)
1. **ml_implementation_status.md** (16K) - Estado atual completo
2. **temporal_split_implementation_plan.md** (38K) - Plano de correção crítica
3. **project_roadmap.md** (6.8K) - Histórico Phases 0-4
4. **historical_comparison_analysis.md** (ESTE DOC - NOVO) - Lições HUB_IA
5. **.github/copilot-instructions.md** - Onboarding AI agents

### 📦 ARQUIVAR em `docs/archive/` (7 docs)
1. **session_checkpoint_phase2_investigation.md** (7.7K) - Checkpoint temporário obsoleto
2. **model_evaluation_report.md** (2.8K) - Duplicado em ml_implementation_status
3. **data_cleaning_report_20251014_144223.md** (730B) - Report específico, pode regenerar
4. **iot-spd-troubleshooting.md** (6.8K) - Troubleshooting específico, não crítico
5. **phase1_feature_expansion.md** (6.8K) - Consolidado em project_roadmap
6. **phase4_instability_detection_plan.md** (9.1K) - Consolidado em project_roadmap
7. **training_comparison_baseline_vs_phase1.md** (9.7K) - Histórico, não essencial

### 📝 ATUALIZAR
1. **.github/copilot-instructions.md** - Adicionar seção "Historical Insights from HUB_IA":
   - Temporal split crítico
   - Lag_diff features importantes
   - Threshold optimization approach
   - GridSearchCV para hyperparameters

---

## 🎯 Roadmap Integrado (Nossa Impl + HUB_IA Insights)

### Phase 5 (ATUAL - CRÍTICO): Temporal Validation
**Duração:** 7-10 horas  
**Baseado em:** HUB_IA Sprint 3 provou viabilidade

- [ ] FASE 1: Fix básico temporal split (1-2h)
- [ ] FASE 2: Cross-validation 5-fold (2-3h)
- [ ] FASE 3: Análise comparativa Random vs Temporal (3-4h)
- [ ] FASE 4: Documentação atualizada (1h)

**Deliverable:** Temporal split funcional, métricas realistas (esperado: 94-97% recall)

---

### Phase 6: Lag Difference Features
**Duração:** 2-3 horas  
**Inspirado em:** HUB_IA `optical_power_lag_diff_6h`

- [ ] Adicionar battery_lag_diff_1d, battery_lag_diff_7d
- [ ] Adicionar rssi_lag_diff_7d, snr_lag_diff_7d
- [ ] Adicionar total_errors_lag_diff_7d
- [ ] Retreinar modelo Phase 6 (22 → 27 features)
- [ ] SHAP analysis para validar importância
- [ ] Comparar performance Phase 4 vs Phase 6

**Deliverable:** +5 features, esperado +2-5% recall em eventos súbitos

---

### Phase 7: Threshold & Hyperparameter Optimization
**Duração:** 3-4 horas  
**Inspirado em:** HUB_IA Precision-Recall curve + GridSearchCV

- [ ] Implementar threshold optimization (Precision-Recall curve)
- [ ] GridSearchCV com TimeSeriesSplit validation
- [ ] Testar XGBoost como alternativa/ensemble
- [ ] Documentar trade-offs (performance vs training time)

**Deliverable:** Modelo otimizado, threshold ajustado para recall target (95%)

---

### Phase 8: Deploy Profissional
**Duração:** 1-2 dias  
**Inspirado em:** HUB_IA branding Lightera

- [ ] Branding EyOn (paleta, logo, CSS)
- [ ] Versionamento de modelos (modelos/*.pkl approach)
- [ ] Health check endpoint
- [ ] Docker deployment
- [ ] CI/CD pipeline

**Deliverable:** Dashboard production-ready com branding corporativo

---

## 🔬 Comparação de Performance (Após Temporal Split)

### Métricas Esperadas

| Métrica | HUB_IA (S3/4) | Nossa - Atual (Random) | Nossa - Esperado (Temporal) | Delta |
|---------|---------------|------------------------|----------------------------|-------|
| **Recall** | "Aceitável" (90%?) | 99.93% | ~94-97% | -3 a -6pp |
| **Precision** | ? | 100% | ~98-100% | -0 a -2pp |
| **False Negatives** | ? | 57 | ~200-400 | +3-7x |
| **False Positives** | ? | 0 | 0-50 | +0 a +50 |
| **Validation** | ✅ Temporal 80/20 | ❌ Random (otimista) | ✅ Temporal 80/20 + CV | Robusto |

**Nota:** Números HUB_IA não especificados nos trechos lidos, precisaria executar notebook completo.

---

## 📖 Referências

### HUB_IA Sprint 3
- **Código:** `historical_docs/HUB_IA_Sprint3/Códigos/`
- **Notebook:** `sensor_ml_model.ipynb` (636 linhas, 59 células)
- **Processamento:** `get_sensor_data.py` (JSON → CSV)
- **Dados:** `dados_sensores.csv` + JSONs Lucas do Rio Verde/Fibrasil

### HUB_IA Sprint 4
- **Código:** `historical_docs/HUB_IA_Sprint4/Entrega - Sprint 4/Códigos/`
- **Deploy:** `app.py` (492 linhas, Streamlit com branding)
- **Modelo:** `dados_estruturados/model_columns.json` (37 features)
- **Assets:** `imagens/` (logo Lightera)

### Projetos Anteriores (Contexto)
- **IFCE:** Notebook experimental, não deployado ❌
- **SENAI:** Baseado em IFCE, 1.5 meses desenvolvimento
- **Confluence:** Pipeline, arquitetura (não detalhado)

---

## 🎓 Conclusão

### O Que Aprendemos de HUB_IA
1. ✅ **Temporal split é CRUCIAL** e viável (prova de conceito validada)
2. ✅ **Lag_diff features** capturam eventos súbitos (mudanças abruptas)
3. ✅ **Threshold optimization** garante recall target realista
4. ✅ **GridSearchCV** pode melhorar +2-3% performance
5. ✅ **Branding profissional** importa para stakeholders

### O Que NOSSA Implementação Inovou
1. ⭐ **Instability detection** (battery CV/range/spikes/trend) - ÚNICA
2. ⭐ **deployment_age_days** - TOP feature (17.5%) - DESCOBERTA
3. ⭐ **SHAP explicabilidade** - HUB_IA não tem
4. ⭐ **Plotly interativo** - Melhor UX que Matplotlib estático
5. ⭐ **4-page structure** - Problem Discovery alinhado com EyOn

### Próximo Passo IMEDIATO
🚨 **Executar temporal_split_implementation_plan.md FASE 1** (1-2 horas)
- Fix básico: preservar @timestamp
- Implementar temporal_train_test_split() com gap 7 dias
- Validar: train dates < test dates

**Após temporal split validado:** Adicionar lag_diff features (Phase 6) inspirados em HUB_IA.

---

**Autor:** AI Agent Context Engineering  
**Data:** 17/Out/2025  
**Fontes:** HUB_IA Sprint 3/4 (Lightera/Fibrasil)  
**Status:** Análise Completa ✅
