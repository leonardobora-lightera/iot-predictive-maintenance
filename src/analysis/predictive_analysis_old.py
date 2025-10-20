import sys
import os

# Workaround for stubborn environment issues: explicitly add local site-packages to path
local_site_packages = "/home/leo/.local/lib/python3.10/site-packages"
if local_site_packages not in sys.path:
    sys.path.insert(0, local_site_packages)

# Add the project root to the Python path to resolve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from src.core.logic import load_data
import os
import matplotlib.pyplot as plt
import matplotlib
try:
    # Ensure a non-interactive backend so SHAP/matplotlib work inside Streamlit
    matplotlib.use('Agg')
except Exception:
    # If setting backend fails, continue — Streamlit may have already configured it
    pass

def safe_shap_force_plot(expected_value, shap_values, feature_row, features_names=None):
    """Render a SHAP force_plot into a matplotlib figure and return it.
    If SHAP isn't available or plotting fails, raise an informative exception.
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError(f"shap is not available: {e}")

    try:
        # Create a matplotlib figure and let shap render into it
        fig, ax = plt.subplots(figsize=(10, 2))

        # Newer SHAP versions (>=0.20) expose shap.plots.force
        if hasattr(shap, 'plots') and hasattr(shap.plots, 'force'):
            # Preferred modern signature: shap.plots.force(expected_value, shap_values, ...)
            try:
                # Create a matplotlib figure and let shap render into it
                fig, ax = plt.subplots(figsize=(10, 2))

                # Helper to normalize inputs and pick single-sample shap vector and base value
                import numpy as _np

                def _ensure_array(x):
                    try:
                        return _np.array(x)
                    except Exception:
                        return _np.array([x])

                sv = shap_values
                base = expected_value

                # If shap_values is a list (multi-class), prefer the positive class (index 1) when available
                if isinstance(sv, (list, tuple)):
                    # pick the last class (commonly the positive class) or index 1 if exists
                    idx = 1 if len(sv) > 1 else -1
                    sv_sel = _ensure_array(sv[idx])
                else:
                    sv_sel = _ensure_array(sv)

                # If sv_sel is 2D (n_samples, n_features), take first sample
                if sv_sel.ndim == 2:
                    sv_sample = sv_sel[0]
                elif sv_sel.ndim == 1:
                    sv_sample = sv_sel
                else:
                    # attempt to squeeze
                    sv_sample = _np.squeeze(sv_sel)

                # Normalize base to scalar if array-like
                if hasattr(base, '__len__') and not isinstance(base, str):
                    try:
                        base_arr = _ensure_array(base)
                        # If base has multiple entries, pick matching class index as above
                        if base_arr.ndim > 0 and base_arr.size > 1:
                            base_val = base_arr[idx] if isinstance(sv, (list, tuple)) or base_arr.size > 1 else float(base_arr)
                        else:
                            base_val = float(base_arr)
                    except Exception:
                        base_val = base
                else:
                    base_val = base


                errors = []

                def _call_and_check(call_fn):
                    # create a fresh figure so we can inspect what was drawn
                    try:
                        plt.close('all')
                    except Exception:
                        pass
                    fig_local = plt.figure(figsize=(10, 2))
                    try:
                        call_fn()
                    except Exception as e:
                        try:
                            plt.close(fig_local)
                        except Exception:
                            pass
                        return False, str(e), None

                    # After call, inspect the current figure for visible content
                    try:
                        fig_after = plt.gcf()
                        has_content = False
                        for ax in fig_after.axes:
                            # check for common drawable artists
                            if len(ax.patches) > 0 or len(ax.lines) > 0 or len(ax.collections) > 0 or len(ax.images) > 0 or len(ax.containers) > 0:
                                has_content = True
                                break
                            # texts other than empty ticks may count
                            if any([t.get_text().strip() for t in ax.texts if t.get_text().strip() != '']):
                                has_content = True
                                break
                        if has_content:
                            return True, None, fig_after
                        else:
                            plt.close(fig_after)
                            return False, 'empty-figure', None
                    except Exception as e:
                        try:
                            plt.close(fig_local)
                        except Exception:
                            pass
                        return False, str(e), None

                # Candidate call patterns to try (prefer matplotlib rendering)
                calls = []

                if hasattr(shap, 'plots') and hasattr(shap.plots, 'force'):
                    # Preferred: enforce matplotlib rendering so a figure is produced
                    calls.append(lambda: shap.plots.force(base_val, shap_values, features=feature_row, matplotlib=True, show=False))
                    calls.append(lambda: shap.plots.force(base_val, sv_sample, features=feature_row, matplotlib=True, show=False))
                    # Multi-output slicing
                    def _call_multi():
                        bv = base_val[0] if hasattr(base_val, '__len__') else base_val
                        svm = shap_values[..., 0] if getattr(shap_values, 'ndim', 0) >= 2 else shap_values
                        shap.plots.force(bv, svm, features=feature_row, matplotlib=True, show=False)
                    calls.append(_call_multi)
                    # As a last resort try JS/HTML variant (may not draw into matplotlib)
                    calls.append(lambda: shap.plots.force(base_val, shap_values))

                # Older API fallback(s) (matplotlib)
                if hasattr(shap, 'force_plot'):
                    calls.append(lambda: shap.force_plot(base_val, shap_values, feature_row, matplotlib=True, show=False))
                    calls.append(lambda: shap.force_plot(base_val, sv_sample, feature_row, matplotlib=True, show=False))

                # Try all candidate calls until one produces visible content
                for c in calls:
                    ok, err, fig_res = _call_and_check(c)
                    if ok and fig_res is not None:
                        plt.tight_layout()
                        return fig_res
                    else:
                        errors.append(err)

                # If all attempts failed or produced empty figures
                raise RuntimeError("shap.plots.force attempts failed or produced empty figures: " + " | ".join([str(x) for x in errors]))

                plt.tight_layout()
                return fig
            except Exception as e:
                plt.close('all')
                raise RuntimeError(f"Failed to render SHAP plot: {e}")
        else:
            raise RuntimeError("No supported SHAP force-plot function found")

        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close('all')
        raise RuntimeError(f"Failed to render SHAP plot: {e}")

st.set_page_config(layout="wide")

@st.cache_data
def load_processed_data():
    """Carrega os dados processados, garantindo que a coluna de timestamp seja datetime."""
    df = load_data()
    if df is not None and '@timestamp' in df.columns:
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
    return df

@st.cache_resource
def load_model():
    """Carrega o modelo preditivo e as features do arquivo."""
    # Constrói o caminho absoluto para o arquivo do modelo na raiz do projeto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_root, 'model.joblib')

    if not os.path.exists(model_path):
        return None, None
    try:
        model_payload = joblib.load(model_path)
        return model_payload['model'], model_payload['features']
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None

def feature_engineering_for_prediction(df):
    """Cria features para um único dispositivo no momento da predição."""
    df = df.sort_values(by=['device_id', '@timestamp']).copy()
    
    # Ensure timestamp is datetime before setting as index
    if '@timestamp' in df.columns:
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
    
    # Define o timestamp como índice para operações de janela de tempo
    df = df.set_index('@timestamp')

    # ========== PHASE 2: RAW FEATURES (keep original values) ==========
    # Battery voltage is already in df, we just need to keep it
    # (it will be preserved through the rolling calculations)
    
    # ========== BATTERY FEATURES ==========
    if 'battery_voltage' in df.columns:
        df['battery_rolling_mean_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').mean().reset_index(0, drop=True)
        df['battery_rolling_std_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').std().reset_index(0, drop=True)

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

    return df



# --- Layout da Aplicação ---
st.title("Análise Preditiva de Falhas de Dispositivos")

model, features = load_model()

if model is None:
    st.warning("Modelo preditivo não encontrado. O treinamento inicial é necessário.")
    st.info("Para treinar o modelo, execute o seguinte comando no seu terminal a partir da raiz do projeto:")
    st.code("python -m src.collector.train_model")
    st.warning("Após a conclusão do treinamento, atualize esta página.")
else:
    st.success("Modelo preditivo carregado com sucesso!")
    data = load_processed_data()

    if data is not None and not data.empty:
        # Filtra para obter apenas dispositivos com dados nos últimos 90 dias
        ninety_days_ago = data['@timestamp'].max() - pd.Timedelta(days=90)
        recent_data = data[data['@timestamp'] >= ninety_days_ago]
        
        if recent_data.empty:
            st.warning("Não há dados de dispositivos nos últimos 90 dias para análise.")
        else:
            available_devices = sorted(recent_data['device_id'].unique())
            device_id = st.selectbox("Selecione o Device ID para a Análise", available_devices)

            if device_id:
                st.header(f"Análise do Dispositivo: {device_id}")
                
                device_data_full = recent_data[recent_data["device_id"] == device_id]

                # Expander para detalhes do dispositivo
                with st.expander("Exibir Detalhes do Dispositivo"):
                    last_record_display = device_data_full.sort_values('@timestamp').iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Última Comunicação", last_record_display['@timestamp'].strftime('%Y-%m-%d %H:%M'))
                    col2.metric("Tensão da Bateria", f"{last_record_display.get('battery_voltage', 0):.2f}V")
                    col3.metric("Versão do Firmware", str(last_record_display.get('fw_app_version', 'N/A')))
                    col4.metric("Qualidade do Sinal", str(last_record_display.get('status_signal_quality', 'N/A')))

                # Lógica de predição
                with st.spinner("Calculando features e fazendo a previsão..."):
                    df_features = feature_engineering_for_prediction(device_data_full)
                    last_record_for_pred = df_features.iloc[[-1]].copy()
                    
                    # Debug: show which features have NaN before filling
                    nan_features = [f for f in features if f in last_record_for_pred.columns and last_record_for_pred[f].isna().any()]
                    if nan_features:
                        st.info(f"⚠️ Features com valores ausentes (preenchidos com 0): {', '.join(nan_features)}")
                    
                    # Fill NaN values in features with 0 (rolling features may be NaN for devices with <7 days of data)
                    # This is a fallback for prediction - in training we drop these rows, but for real-time prediction
                    # we need to make a prediction even with incomplete rolling window data
                    for feature in features:
                        if feature in last_record_for_pred.columns:
                            last_record_for_pred[feature] = last_record_for_pred[feature].fillna(0)

                # Check if we have the required features
                missing_features = [f for f in features if f not in last_record_for_pred.columns]
                if missing_features:
                    st.warning(f"O dispositivo selecionado não possui as seguintes features necessárias: {missing_features}")
                else:
                    st.subheader("Previsão de Falha")
                    prediction_proba = model.predict_proba(last_record_for_pred[features])
                    failure_probability = prediction_proba[0][1]

                    # Gauge Chart para Probabilidade de Falha
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=failure_probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risco de Falha (%)", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        number={'suffix': "%", 'font': {'size': 40}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': 'lightgreen'},
                                {'range': [50, 80], 'color': 'orange'},
                                {'range': [80, 100], 'color': 'red'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': failure_probability * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Interpretação do risco
                    if failure_probability < 0.3:
                        st.success(f"✅ **Baixo risco de falha:** {failure_probability*100:.2f}% - Dispositivo operando normalmente")
                    elif failure_probability < 0.7:
                        st.warning(f"⚠️ **Risco moderado de falha:** {failure_probability*100:.2f}% - Monitorar o dispositivo")
                    else:
                        st.error(f"🚨 **Alto risco de falha:** {failure_probability*100:.2f}% - Intervenção recomendada!")

                    # SHAP explainability section
                    with st.expander("Explicar Previsão (SHAP)"):
                        try:
                            import shap
                            # Use TreeExplainer for tree-based models (RandomForest)
                            explainer = shap.TreeExplainer(model)
                            # raw shap_values for the sample (may be list or ndarray)
                            raw_shap = explainer.shap_values(last_record_for_pred[features])

                            # Determine class choices for user (if applicable)
                            class_index = None
                            class_options = None
                            # expected_value may be scalar or array per class
                            expected_raw = explainer.expected_value if hasattr(explainer, 'expected_value') else None

                            # Prepare UI selector when multiple classes available
                            if isinstance(raw_shap, (list, tuple)) and len(raw_shap) > 1:
                                # binary/multiclass as list of arrays
                                class_options = [f"class_{i}" for i in range(len(raw_shap))]
                                # default to last class (commonly positive)
                                choice = st.selectbox("Escolha a classe a explicar", options=class_options, index=len(class_options)-1)
                                class_index = int(choice.split('_')[-1])
                            else:
                                # ndarray: check last axis
                                import numpy as _np
                                try:
                                    arr = _np.asarray(raw_shap)
                                    if arr.ndim == 3:
                                        # shape (n_samples, n_features, n_classes)
                                        n_classes = arr.shape[-1]
                                        class_options = [f"class_{i}" for i in range(n_classes)]
                                        choice = st.selectbox("Escolha a classe a explicar", options=class_options, index=n_classes-1)
                                        class_index = int(choice.split('_')[-1])
                                    else:
                                        class_index = None
                                except Exception:
                                    class_index = None

                            # Helper to extract 1D shap vector for the chosen class
                            def _pick_shap_vector(sv, cls_idx=None):
                                import numpy as _np
                                if isinstance(sv, (list, tuple)):
                                    idx = cls_idx if cls_idx is not None else (len(sv)-1)
                                    sel = _np.asarray(sv[idx])
                                    # sel may be (n_samples, n_features)
                                    if sel.ndim == 2:
                                        return sel[0]
                                    return _np.ravel(sel)
                                else:
                                    arr = _np.asarray(sv)
                                    if arr.ndim == 3:
                                        idx = cls_idx if cls_idx is not None else (arr.shape[-1]-1)
                                        return arr[0, :, idx]
                                    if arr.ndim == 2:
                                        return arr[0]
                                    return _np.ravel(arr)

                            try:
                                shap_vec = _pick_shap_vector(raw_shap, class_index)
                            except Exception as e:
                                st.error(f"Falha ao normalizar shap_values: {e}")
                                shap_vec = None

                            # Determine expected/base value for chosen class
                            def _pick_expected(ev, cls_idx=None):
                                import numpy as _np
                                if ev is None:
                                    return None
                                if isinstance(ev, (list, tuple)):
                                    idx = cls_idx if cls_idx is not None else (len(ev)-1)
                                    return float(ev[idx])
                                arr = _np.asarray(ev)
                                if arr.ndim > 0 and arr.size > 1:
                                    idx = cls_idx if cls_idx is not None else (arr.size-1)
                                    return float(arr[idx])
                                return float(arr)

                            expected_val = _pick_expected(expected_raw, class_index)

                            # Show a compact summary: base, sum(shap), predicted prob
                            if shap_vec is not None and expected_val is not None:
                                import numpy as _np
                                sum_shap = float(_np.sum(shap_vec))
                                predicted = expected_val + sum_shap
                                # Clamp predicted probability for display
                                try:
                                    predicted_display = float(predicted)
                                    if predicted_display != predicted_display:  # NaN
                                        predicted_display = 0.0
                                except Exception:
                                    predicted_display = 0.0
                                predicted_display = max(0.0, min(1.0, predicted_display))
                                c1, c2, c3 = st.columns([1,1,1])
                                c1.metric("Expected (base)", f"{expected_val:.4f}")
                                c2.metric("Sum SHAP", f"{sum_shap:+.4f}")
                                c3.metric("Predicted (base+sum)", f"{predicted_display:.4f}")

                                # Short cheat-sheet for non-experts
                                st.markdown("**O que isto significa:** 'Expected' é o ponto de partida do modelo; 'Sum SHAP' mostra quanto as features empurraram a previsão; 'Predicted' é o resultado (valores entre 0 e 1).\n\nSe 'Predicted' estiver em 0, as features reduziram a probabilidade praticamente a zero.")

                            # Try to render SHAP force plot via safe helper; if it fails, show fallback bar chart
                            if shap_vec is not None:
                                try:
                                    # safe_shap_force_plot expects expected_value and shap_values (can be 1D)
                                    fig_shap = safe_shap_force_plot(expected_val, shap_vec, last_record_for_pred[features].iloc[0], features_names=features)
                                    st.pyplot(fig_shap)
                                    plt.close(fig_shap)
                                except Exception as e:
                                    st.warning(f"SHAP force plot falhou ou não está disponível: {e}")
                                # Always show a reliable fallback bar chart (helps ensure UX)
                                try:
                                    import matplotlib.pyplot as _plt
                                    import numpy as _np
                                    vals = _np.array(shap_vec).flatten()
                                    feat_names = features
                                    # order by absolute contribution
                                    order = _np.argsort(_np.abs(vals))
                                    fig_f, ax_f = _plt.subplots(figsize=(6, max(2, len(vals)*0.4)))
                                    colors = [ '#d7191c' if v<0 else '#2b83ba' for v in vals[order] ]
                                    ax_f.barh(range(len(vals)), vals[order], color=colors)
                                    ax_f.set_yticks(range(len(vals)))
                                    ax_f.set_yticklabels([feat_names[i] for i in order])
                                    ax_f.axvline(0, color='k', linewidth=0.6)
                                    ax_f.set_xlabel('SHAP value (classe escolhida)')
                                    fig_f.tight_layout()
                                    st.markdown('**Fallback (contribuições por feature):**')
                                    st.pyplot(fig_f)
                                    _plt.close(fig_f)
                                except Exception as e2:
                                    st.error(f"Falha ao gerar fallback gráfico: {e2}")
                        except Exception as e:
                            st.error(f"SHAP não está disponível ou falhou ao calcular: {e}")


    else:
        st.error("Não foi possível carregar os dados processados. Verifique o arquivo 'payloads_processed.csv'.")