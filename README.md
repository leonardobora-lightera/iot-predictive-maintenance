# 🔋 IoT Battery Predictive Maintenance POC

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.48%2B-green)](https://shap.readthedocs.io/)

**Proof of Concept (POC)** para detecção de **instabilidade de bateria** em sensores IoT antes da falha total, usando Random Forest e features de oscilação.

---

## 🎯 Problema

Sensores IoT em campo apresentam falhas de bateria sem warning prévio. Thresholds fixos de voltagem (ex: `battery_voltage < 2.5V`) só detectam falhas **já ocorridas**, não permitindo manutenção preventiva.

### Inovação Proposta
Detectar **padrões de instabilidade** (oscilações, variações, picos) 7-30 dias antes da falha total através de features de janela temporal:
- `battery_range_7d`: Amplitude de oscilação em 7 dias
- `battery_cv_7d`: Coeficiente de variação (instabilidade relativa)
- `battery_spike_count_7d`: Contagem de picos abruptos

---

## 🚀 O que foi Construído

### ✅ 1. Data Engineering Pipeline
- **Outlier Detection**: Remove erros de sensor (battery >5V, signal fora de spec LoRa/LTE)
- **Clean Dataset**: 417,022 samples processados (9.6% battery outliers + 26.9% signal outliers removidos)
- **Tool**: `tools/clean_data.py` com relatório detalhado de limpeza

### ✅ 2. Feature Innovation
**22 features** em 4 categorias (Phase 4):

| Categoria | Features | Destaque |
|-----------|----------|----------|
| **Battery (10)** | voltage, rolling mean/std 7d, critical thresholds, **instability metrics** | `battery_range_7d` (8.01% importance) |
| **Signal (4)** | RSSI/SNR rolling mean/std 7d | Correlação fraca com falha battery |
| **Communication (3)** | f_cnt, errors rolling sum 7d | Testado mas não preditivo |
| **Context (2)** | deployment_age_days, msg_error_pct | Features auxiliares |

**Validação SHAP**: Features de instabilidade (`battery_range_7d`, `battery_cv_7d`) são 61x mais preditivas que thresholds fixos.

### ✅ 3. Dashboard Streamlit
Interface profissional com 4 páginas:

1. **📊 Overview**: Métricas da frota, high-risk alerts
2. **🎯 Análise de Dispositivo**: Seleção por device, gauge de risco (0-100%), SHAP force plots, histórico de bateria
3. **🧬 Jornada do Projeto**: Evolução Phase 0→4 com rationale
4. **📈 Insights & Top Riscos**: (Placeholder para expansão futura)

**Demo**: `streamlit run src/analysis/predictive_analysis.py`

### ✅ 4. Model Explainability
- **SHAP Integration**: TreeExplainer para feature importance e force plots por dispositivo
- **Fallback Robusto**: Usa `model.feature_importances_` se SHAP falha
- **Visualização**: Top 10 features com impacto direcional

---

## 📊 Jornada Metodológica

### Phase 4: Instability Detection (22 features)
**Resultados iniciais**: 99.93% recall em 80/20 split (aleatório)

**Problema descoberto**: Métricas infladas por data leakage (split não temporal)

---

### Phase 6: Feature Expansion (27 features)
**Objetivo**: Melhorar recall adicionando:
- Signal features (RSSI/SNR)
- Communication features (f_cnt, errors)
- **Lag_diff features** (capturar mudanças abruptas)
- **Threshold optimization** (Precision-Recall curve)

**Resultado**: 85.06% recall em 80/20 split temporal

**PROBLEMA CRÍTICO descoberto por Cross-Validation**:
```
5-Fold TimeSeriesSplit Results:
- Fold 1: 12.26% recall
- Fold 2: 27.27% recall
- Fold 3: 42.19% recall
- Fold 4: 78.26% recall
- Fold 5: 88.29% recall

Mean: 49.65% ± 30.44%
CV (Coefficient of Variation): 61.4% 🔴
```

**Learning Crítico**: Single 80/20 split pode ser **altamente enganoso**. Cross-validation revelou instabilidade temporal severa.

---

### Strategic Regression: Phase 4.5 Battery-Centric
**Decisão**: Regredir para baseline focado APENAS em bateria (8-9 features).

**Rationale**:
1. **Feature Dilution**: Signal/comm features não correlacionados com falha battery
2. **Temporal Overfitting**: Lag_diff features funcionam em períodos recentes, falham em períodos antigos
3. **Complexidade**: 27 features aumentaram variance do modelo sem ganho de bias
4. **HUB_IA Learnings**: Projetos de sucesso têm foco específico (óptica, não genérico)

**Features Phase 4.5** (8-9 battery-only):
- `battery_voltage` (baseline)
- `battery_rolling_mean_7d`, `battery_rolling_std_7d` (tendência)
- `battery_range_7d`, `battery_cv_7d`, `battery_spike_count_7d` (instabilidade)
- `battery_critical`, `battery_very_low` (thresholds)
- `battery_drop_rate_7d` (taxa de degradação)

**Status**: Baseline definido, documentado em `docs/phase45_regression_rationale.md` (22KB). **Implementação planejada pós-apresentação**.

---

## 🔬 Learnings Metodológicos

### 1. Cross-Validation is Critical
❌ **Antes**: Confiar em single 80/20 split (99.93% recall → enganoso)  
✅ **Depois**: 5-fold TimeSeriesSplit como quality gate (descobriu CV=61.4%)

### 2. Feature Dilution é Real
❌ **Antes**: "Mais features = melhor modelo"  
✅ **Depois**: Features não correlacionadas aumentam instabilidade temporal

### 3. Strategic Regression > Sunk Cost Fallacy
❌ **Antes**: Insistir em Phase 6 porque "já investimos tempo"  
✅ **Depois**: Regredir para baseline quando evidências mostram instabilidade

### 4. Instability Detection Validated
✅ **Innovation**: `battery_range_7d` (8.01% importance via SHAP)  
✅ **Value**: 61x mais preditivo que thresholds fixos  
✅ **Replicable**: Pattern pode ser aplicado a outros sensores IoT

---

## 🛠️ Setup & Usage

### Pré-requisitos
```bash
# Python 3.10+
pip install -r requirements.txt
```

### 1. Data Cleaning
```bash
python tools/clean_data.py --report
# Output: data/processed/payloads_processed_clean.csv
#         docs/data_cleaning_report_YYYYMMDD_HHMMSS.md
```

### 2. Model Training (Phase 4)
```bash
python tools/train_monitor.py --days 90 --preserve-model
# Output: model.joblib (root)
#         train_monitor_output/model_YYYYMMDD_HHMMSS.joblib
#         train_monitor_output/train_report_*.json
```

### 3. Dashboard
```bash
streamlit run src/analysis/predictive_analysis.py
# Access: http://localhost:8501
```

### 4. SHAP Analysis
```bash
python tools/shap_analysis.py
# Requires: model.joblib exists
# Output: shap_analysis_output/feature_importance_*.csv
```

---

## 📁 Estrutura do Projeto

```
.
├── data/
│   ├── raw/                    # CSVs originais (não versionados)
│   └── processed/              # Dados limpos (não versionados)
├── src/
│   ├── analysis/
│   │   └── predictive_analysis.py  # Dashboard Streamlit
│   ├── collector/
│   │   └── train_model.py      # Pipeline de treino
│   └── core/
│       └── logic.py            # Data loading com config.ini
├── tools/
│   ├── clean_data.py           # Outlier removal
│   ├── train_monitor.py        # Resource monitoring wrapper
│   └── shap_analysis.py        # Feature importance
├── docs/
│   ├── phase45_regression_rationale.md  # Strategic decision
│   ├── historical_comparison_analysis.md # HUB_IA learnings
│   └── CHECKPOINT_PHASE45.md   # Current baseline
├── config.ini                  # Paths configuration
└── requirements.txt
```

---

## 🔮 Roadmap (Pós-Apresentação)

### Phase 4.5: Battery-Centric Baseline
**Prioridade**: CRÍTICO  
**Estimativa**: 5-7 horas  
**Objetivo**: Validar hipótese de que 8-9 battery features são suficientes e estáveis

**Tasks**:
- [ ] Simplificar `train_model.py` para 8-9 features battery-only
- [ ] Corrigir temporal split (preservar `@timestamp` antes dropna)
- [ ] Treinar + threshold optimization (target recall ≥85%)
- [ ] Cross-validation 5-fold: Target CV<20% (ideal), CV<30% (mínimo)
- [ ] Documentar resultados: `docs/phase45_battery_centric_results.md`

**Success Criteria**:
- Mean recall ≥70%
- CV <20% (demonstra estabilidade temporal)
- Per-fold recall ≥60% (sem outliers extremos como Phase 6)

### Phase 5: Temporal Validation
- Implementar `TimeSeriesSplit` com gap de 7 dias
- Validar que test set é completamente FUTURO em relação a train
- Comparar métricas com/sem temporal split

### Phase 6 (Opcional): Incremental Features
- **SE** Phase 4.5 estável (CV<20%):
  - Testar adicionar signal features (RSSI, SNR) uma de cada vez
  - Validar se melhoram recall sem degradar estabilidade
  - Cross-validation após cada adição

---

## 📚 Documentação Técnica

- **ML Status**: `docs/ml_implementation_status.md` - Estado atual completo
- **Regression Rationale**: `docs/phase45_regression_rationale.md` - Por que regredir
- **HUB_IA Learnings**: `docs/historical_comparison_analysis.md` - Padrões validados
- **Checkpoint**: `docs/CHECKPOINT_PHASE45.md` - Baseline atual

---

## 🎓 HUB_IA Context Engineering

Este projeto aplicou **learnings validados** do HUB_IA Sprint 3/4:

| Learning | HUB_IA Original | Aplicação Neste POC |
|----------|-----------------|---------------------|
| **Temporal Split** | `df['time'].quantile(0.8)` | Implementado em Phase 6, descobriu instabilidade |
| **Lag Diff Features** | `optical_power_lag_diff_6h` | Testado em Phase 6, causou overfitting temporal |
| **Threshold Optimization** | P-R curve, target recall≥90% | Implementado, mas métricas não generalizaram |
| **Cross-Validation** | TimeSeriesSplit 5-fold | **Quality gate** que salvou deploy de modelo instável |

**Key Takeaway**: Context engineering funciona quando adaptado ao problema específico (battery health ≠ optical power).

---

## ⚠️ Limitações Conhecidas

1. **Temporal Split Bug**: `@timestamp` dropado durante `dropna()`, causando leakage em Phase 4
2. **Phase 4.5 Not Implemented**: Baseline definido mas não executado (planejado pós-apresentação)
3. **Model Instability**: Phase 6 (27 features) não production-ready (CV=61.4%)
4. **Dataset Size**: 417k samples podem ter concept drift temporal (não investigado)
5. **Feature Parity**: Dashboard `feature_engineering_for_prediction()` deve sempre match `train_model.py`

---

## 🤝 Contribuindo

Este é um POC em desenvolvimento. Próximos passos:

1. Implementar Phase 4.5 baseline
2. Validar estabilidade temporal (CV<20%)
3. Decidir sobre incremental features ou production deployment

**Feedback bem-vindo** sobre metodologia, feature engineering, ou temporal validation.

---

## 📄 Licença

MIT License - Ver LICENSE file para detalhes

---

## 👤 Autor

**Leonardo Bora**  
Software Engineering Intern @ Lightera  
📧 leonardo.costa@lightera.com  
🔗 [GitHub](https://github.com/leonardobora-lightera)

---

**Status**: Phase 4.5 Baseline Planned | Presentation Ready | Methodologically Solid POC

*Last Updated: October 20, 2025*
