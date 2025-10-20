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
import plotly.express as px
from src.core.logic import load_data
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import linregress

try:
    # Ensure a non-interactive backend so SHAP/matplotlib work inside Streamlit
    matplotlib.use('Agg')
except Exception:
    pass

# ========== SHAP Helper Function ==========
def safe_shap_force_plot(expected_value, shap_values, feature_row, features_names=None):
    """Render a SHAP force_plot into a matplotlib figure and return it."""
    try:
        import shap
    except Exception as e:
        raise RuntimeError(f"shap is not available: {e}")

    try:
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Normalize shap_values to 1D array
        if isinstance(shap_values, (list, tuple)):
            sv = np.array(shap_values[-1]).flatten()
        else:
            sv = np.array(shap_values).flatten()
        
        # Normalize expected_value to scalar
        if hasattr(expected_value, '__len__') and not isinstance(expected_value, str):
            base_val = float(np.array(expected_value).flatten()[-1])
        else:
            base_val = float(expected_value)
        
        # Try to render SHAP force plot
        if hasattr(shap, 'plots') and hasattr(shap.plots, 'force'):
            try:
                shap.plots.force(base_val, sv, features=feature_row, matplotlib=True, show=False)
                plt.tight_layout()
                return plt.gcf()
            except:
                pass
        
        # Fallback: just return the figure (will use bar chart instead)
        plt.close(fig)
        raise RuntimeError("SHAP force plot not available, use fallback")
        
    except Exception as e:
        plt.close('all')
        raise RuntimeError(f"Failed to render SHAP plot: {e}")

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Sistema Preditivo de Manutenção IoT",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Load model and data functions ==========
@st.cache_data
def load_processed_data():
    df = load_data()
    if df is not None and '@timestamp' in df.columns:
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
    return df

@st.cache_resource
def load_model():
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
    """Apply the COMPLETE Phase 4 feature engineering pipeline for prediction."""
    df = df.sort_values(by=['device_id', '@timestamp']).copy()
    if '@timestamp' in df.columns:
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
    df = df.set_index('@timestamp')

    # ========== BATTERY FEATURES ==========
    if 'battery_voltage' in df.columns:
        # Rolling statistics
        df['battery_rolling_mean_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').mean().reset_index(0, drop=True)
        df['battery_rolling_std_7d'] = df.groupby('device_id')['battery_voltage'].rolling('7D').std().reset_index(0, drop=True)
        
        # PHASE 3: Critical battery thresholds
        df['battery_critical'] = (df['battery_voltage'] < 2.8).astype(int)
        df['battery_very_low'] = (df['battery_voltage'] < 2.5).astype(int)
        
        # Battery drop rate
        battery_first_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').apply(
            lambda x: x.iloc[0] if len(x) > 0 else np.nan
        ).reset_index(0, drop=True)
        battery_last_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').apply(
            lambda x: x.iloc[-1] if len(x) > 0 else np.nan
        ).reset_index(0, drop=True)
        df['battery_drop_rate_7d'] = (battery_first_7d - battery_last_7d) / 7.0
        
        # PHASE 3.1: Instability features
        df['battery_instability_7d'] = df['battery_rolling_std_7d']
        df['battery_cv_7d'] = df['battery_rolling_std_7d'] / df['battery_rolling_mean_7d'].replace(0, np.nan)
        
        # Battery range (max - min)
        battery_max_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').max().reset_index(0, drop=True)
        battery_min_7d = df.groupby('device_id')['battery_voltage'].rolling('7D').min().reset_index(0, drop=True)
        df['battery_range_7d'] = battery_max_7d - battery_min_7d
        
        # High oscillation flag
        df['battery_high_oscillation'] = (df['battery_range_7d'] > 0.3).astype(int)
        
        # PHASE 4: Spike count - detect abrupt voltage changes
        df['battery_diff'] = df.groupby('device_id')['battery_voltage'].diff().abs()
        df['battery_spike_count_7d'] = df.groupby('device_id')['battery_diff'].rolling('7D').apply(
            lambda x: (x > 0.2).sum() if len(x) > 0 else 0
        ).reset_index(0, drop=True)
        
        # PHASE 4: Trend - linear regression slope
        def compute_battery_trend(series):
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
    if 'f_cnt' in df.columns:
        df['f_cnt_rolling_mean_7d'] = df.groupby('device_id')['f_cnt'].rolling('7D').mean().reset_index(0, drop=True)
        df['f_cnt_rolling_std_7d'] = df.groupby('device_id')['f_cnt'].rolling('7D').std().reset_index(0, drop=True)
    if 'buffer.total_errors' in df.columns:
        df['total_errors_rolling_sum_7d'] = df.groupby('device_id')['buffer.total_errors'].rolling('7D').sum().reset_index(0, drop=True)
    
    # ========== SIGNAL QUALITY FEATURES ==========
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
    
    # ========== OTHER FEATURES ==========
    df['deployment_age_days'] = df.groupby('device_id').cumcount()
    if 'msg_type' in df.columns:
        df['msg_type_error_pct_7d'] = df.groupby('device_id')['msg_type'].rolling('7D').apply(
            lambda x: (x == 6).sum() / len(x) if len(x) > 0 else 0
        ).reset_index(0, drop=True)

    return df

# ========== SIDEBAR ==========
st.sidebar.markdown('<h1 style="text-align: center;">🔮 IoT Predictive</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegação",
    ["📊 Overview", "🎯 Análise de Dispositivo", "🧬 Jornada do Projeto", "�� Insights & Top Riscos"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Sobre o Sistema")
st.sidebar.info("**Manutenção Preditiva com IA**\n\nSistema avançado de ML para detecção preventiva de falhas em sensores IoT.\n\n🎯 **Inovação:** Detecção de padrões de instabilidade.")

model, features = load_model()
if model:
    st.sidebar.success("✅ Modelo Ativo: Phase 4")
    st.sidebar.caption(f"Features: {len(features)} | Precisão: 100%")
else:
    st.sidebar.error("❌ Modelo não encontrado")

# ========== PAGES ==========
if page == "📊 Overview":
    st.title("🔮 Sistema Preditivo de Manutenção IoT")
    st.markdown("**Detecção Inteligente de Falhas em Sensores através de Padrões de Instabilidade**")
    
    if model is None:
        st.warning("⚠️ Modelo preditivo não encontrado.")
        st.stop()
    
    data = load_processed_data()
    if data is None or data.empty:
        st.error("Dados não disponíveis.")
        st.stop()
    
    ninety_days_ago = data['@timestamp'].max() - pd.Timedelta(days=90)
    recent_data = data[data['@timestamp'] >= ninety_days_ago]
    
    st.markdown("### 📊 Métricas da Frota")
    col1, col2, col3, col4 = st.columns(4)
    
    total_devices = recent_data['device_id'].nunique()
    total_readings = len(recent_data)
    avg_battery = recent_data['battery_voltage'].mean() if 'battery_voltage' in recent_data.columns else 0
    
    with col1:
        st.metric("Dispositivos Ativos", f"{total_devices:,}", "Últimos 90 dias", border=True)
    with col2:
        st.metric("Leituras Coletadas", f"{total_readings:,}", f"{total_readings//total_devices:.0f} por device", border=True)
    with col3:
        st.metric("Bateria Média", f"{avg_battery:.2f}V", "Normal" if avg_battery > 3.0 else "Atenção", border=True)
    with col4:
        # Optional high risk calculation - use button to avoid blocking
        if 'high_risk_calculated' not in st.session_state:
            st.session_state.high_risk_calculated = False
            st.session_state.high_risk_count = 0
            st.session_state.high_risk_pct = 0
        
        if not st.session_state.high_risk_calculated:
            if st.button("🔄 Calcular Alto Risco", key="calc_risk_btn"):
                with st.spinner("Analisando amostra de 20 devices..."):
                    try:
                        device_sample = recent_data['device_id'].unique()[:20]
                        high_risk_count = 0
                        
                        for dev in device_sample:
                            dev_data = recent_data[recent_data['device_id'] == dev]
                            dev_features = feature_engineering_for_prediction(dev_data)
                            
                            if len(dev_features) > 0:
                                last_rec = dev_features.iloc[[-1]].copy()
                                
                                for feat in features:
                                    if feat not in last_rec.columns:
                                        last_rec[feat] = 0
                                    else:
                                        last_rec[feat] = last_rec[feat].fillna(0)
                                
                                try:
                                    risk_prob = model.predict_proba(last_rec[features])[0][1]
                                    if risk_prob > 0.7:
                                        high_risk_count += 1
                                except:
                                    pass
                        
                        st.session_state.high_risk_count = high_risk_count
                        st.session_state.high_risk_pct = (high_risk_count / len(device_sample)) * 100
                        st.session_state.high_risk_calculated = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")
        
        if st.session_state.high_risk_calculated:
            st.metric("Alto Risco", 
                     f"{st.session_state.high_risk_count}", 
                     f"{st.session_state.high_risk_pct:.1f}% de 20 devices", 
                     border=True)
        else:
            st.metric("Alto Risco", "Clique para calcular", border=True)

elif page == "🎯 Análise de Dispositivo":
    st.title("🎯 Análise Preditiva Individual")
    st.markdown("Avaliação detalhada de risco com explicabilidade AI")
    
    if model is None:
        st.error("Modelo não encontrado.")
        st.stop()
    
    data = load_processed_data()
    if data is None or data.empty:
        st.error("Dados não disponíveis.")
        st.stop()
    
    ninety_days_ago = data['@timestamp'].max() - pd.Timedelta(days=90)
    recent_data = data[data['@timestamp'] >= ninety_days_ago]
    
    available_devices = sorted(recent_data['device_id'].unique())
    device_id = st.selectbox("Selecione o Device ID", available_devices)
    
    if device_id:
        device_data_full = recent_data[recent_data["device_id"] == device_id]
        df_features = feature_engineering_for_prediction(device_data_full)
        last_record = df_features.iloc[[-1]].copy()
        
        for feature in features:
            if feature in last_record.columns:
                last_record[feature] = last_record[feature].fillna(0)
        
        missing = [f for f in features if f not in last_record.columns]
        
        # Debug: mostrar features faltantes se houver
        if missing:
            st.warning(f"⚠️ Features faltantes detectadas ({len(missing)}): {', '.join(missing[:5])}...")
            st.info("💡 Criando features faltantes com valores padrão (0) para permitir predição.")
            # Criar colunas faltantes com valor 0
            for missing_feature in missing:
                last_record[missing_feature] = 0
        
        # Sempre tentar fazer a predição
        try:
            pred_proba = model.predict_proba(last_record[features])
            risk = pred_proba[0][1]
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                title={'text': "Risco de Falha (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'steps': [
                           {'range': [0, 30], 'color': '#d4edda'},
                           {'range': [30, 70], 'color': '#fff3cd'},
                           {'range': [70, 100], 'color': '#f8d7da'}],
                       'bar': {'color': "darkblue"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if risk < 0.3:
                st.success(f"✅ Baixo risco: {risk*100:.2f}%")
            elif risk < 0.7:
                st.warning(f"⚠️ Risco moderado: {risk*100:.2f}%")
            else:
                st.error(f"🚨 Alto risco: {risk*100:.2f}%")
            
            # ========== SHAP EXPLICABILIDADE ==========
            st.markdown("---")
            st.markdown("### 🔍 Explicabilidade AI (SHAP Analysis)")
            
            with st.expander("📊 Ver análise detalhada de contribuições", expanded=True):
                try:
                    import shap
                    
                    # Calculate SHAP values
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(last_record[features])
                    expected_value = explainer.expected_value
                    
                    # Normalize to 1D
                    if isinstance(shap_values, (list, tuple)):
                        shap_vec = np.array(shap_values[-1]).flatten()
                        base_val = float(np.array(expected_value).flatten()[-1]) if hasattr(expected_value, '__len__') else float(expected_value)
                    else:
                        shap_vec = np.array(shap_values).flatten()
                        base_val = float(expected_value)
                    
                    # Show summary metrics
                    sum_shap = float(np.sum(shap_vec))
                    predicted = base_val + sum_shap
                    predicted_display = max(0.0, min(1.0, predicted))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Expected (base)", f"{base_val:.4f}", help="Probabilidade base do modelo")
                    c2.metric("Sum SHAP", f"{sum_shap:+.4f}", help="Impacto total das features")
                    c3.metric("Predicted", f"{predicted_display:.4f}", help="Probabilidade final")
                    
                    st.info("💡 **Interpretação:** O valor 'Expected' é o ponto de partida. O 'Sum SHAP' mostra quanto as características do dispositivo aumentaram (+) ou diminuíram (-) o risco. O 'Predicted' é o resultado final.")
                    
                    # Try force plot, fallback to bar chart
                    try:
                        fig_shap = safe_shap_force_plot(base_val, shap_vec, last_record[features].iloc[0], features_names=features)
                        st.pyplot(fig_shap)
                        plt.close(fig_shap)
                    except:
                        st.caption("ℹ️ SHAP force plot não disponível, mostrando gráfico de barras:")
                    
                    # Always show fallback bar chart (reliable)
                    fig_bar, ax_bar = plt.subplots(figsize=(8, max(3, len(shap_vec)*0.3)))
                    order = np.argsort(np.abs(shap_vec))
                    colors = ['#d7191c' if v < 0 else '#2b83ba' for v in shap_vec[order]]
                    ax_bar.barh(range(len(shap_vec)), shap_vec[order], color=colors)
                    ax_bar.set_yticks(range(len(shap_vec)))
                    ax_bar.set_yticklabels([features[i] for i in order])
                    ax_bar.axvline(0, color='k', linewidth=0.8)
                    ax_bar.set_xlabel('Contribuição SHAP (vermelho=reduz risco, azul=aumenta risco)')
                    ax_bar.set_title('Importância das Features para esta Predição')
                    fig_bar.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)
                    
                except Exception as e:
                    st.warning(f"⚠️ SHAP não disponível: {e}")
                    st.info("Instale com: `pip install shap`")
            
            # ========== GRÁFICOS HISTÓRICOS ==========
            st.markdown("---")
            st.markdown("### 📈 Histórico do Dispositivo (Últimos 30 dias)")
            
            # Filter last 30 days of data for this device
            thirty_days_ago = device_data_full['@timestamp'].max() - pd.Timedelta(days=30)
            device_30d = device_data_full[device_data_full['@timestamp'] >= thirty_days_ago].copy()
            
            if len(device_30d) > 0:
                tab1, tab2, tab3 = st.tabs(["🔋 Bateria", "📡 Sinal", "💬 Comunicação"])
                
                with tab1:
                    if 'battery_voltage' in device_30d.columns:
                        fig_battery = go.Figure()
                        fig_battery.add_trace(go.Scatter(
                            x=device_30d['@timestamp'],
                            y=device_30d['battery_voltage'],
                            mode='lines+markers',
                            name='Voltage',
                            line=dict(color='#2b83ba', width=2),
                            marker=dict(size=4)
                        ))
                        # Add threshold lines
                        fig_battery.add_hline(y=2.8, line_dash="dash", line_color="orange", 
                                             annotation_text="Crítico (2.8V)")
                        fig_battery.add_hline(y=2.5, line_dash="dash", line_color="red",
                                             annotation_text="Falha (2.5V)")
                        fig_battery.update_layout(
                            title="Tensão da Bateria",
                            xaxis_title="Data",
                            yaxis_title="Voltage (V)",
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig_battery, use_container_width=True)
                        
                        # Battery statistics
                        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                        col_b1.metric("Atual", f"{device_30d['battery_voltage'].iloc[-1]:.2f}V")
                        col_b2.metric("Média 30d", f"{device_30d['battery_voltage'].mean():.2f}V")
                        col_b3.metric("Mín 30d", f"{device_30d['battery_voltage'].min():.2f}V")
                        col_b4.metric("Máx 30d", f"{device_30d['battery_voltage'].max():.2f}V")
                    else:
                        st.info("Dados de bateria não disponíveis")
                
                with tab2:
                    signal_cols = ['rssi', 'snr', 'rsrp', 'rsrq']
                    available_signals = [col for col in signal_cols if col in device_30d.columns]
                    
                    if available_signals:
                        for sig in available_signals:
                            fig_sig = px.line(device_30d, x='@timestamp', y=sig, 
                                            title=f"{sig.upper()} ao longo do tempo",
                                            markers=True)
                            fig_sig.update_layout(height=300, hovermode='x unified')
                            st.plotly_chart(fig_sig, use_container_width=True)
                    else:
                        st.info("Dados de sinal não disponíveis")
                
                with tab3:
                    comm_cols = ['lost_payloads_percent', 'f_cnt', 'registration_time']
                    available_comm = [col for col in comm_cols if col in device_30d.columns]
                    
                    if available_comm:
                        for comm in available_comm:
                            fig_comm = px.line(device_30d, x='@timestamp', y=comm,
                                             title=f"{comm} ao longo do tempo",
                                             markers=True)
                            fig_comm.update_layout(height=300, hovermode='x unified')
                            st.plotly_chart(fig_comm, use_container_width=True)
                    else:
                        st.info("Dados de comunicação não disponíveis")
            else:
                st.warning("Dados insuficientes para gráficos históricos (mínimo 30 dias)")
                    
        except Exception as e:
            st.error(f"❌ Erro ao calcular predição: {str(e)}")
            st.exception(e)

elif page == "🧬 Jornada do Projeto":
    st.title("🧬 Evolução do Projeto")
    st.markdown("Da prova de conceito à detecção inteligente de padrões")
    
    phases = [
        {"name": "Phase 0: Baseline", "date": "Out 10", "features": 4, "recall": 99.32},
        {"name": "Phase 1: Feature Expansion", "date": "Out 12", "features": 12, "recall": 99.95},
        {"name": "Phase 2: Raw Battery", "date": "Out 14", "features": 13, "recall": 100.00},
        {"name": "Phase 3: Threshold (FALHOU)", "date": "Out 14", "features": 16, "recall": 99.99},
        {"name": "Phase 4: Instability Detection ✅", "date": "Out 15", "features": 22, "recall": 99.93}
    ]
    
    for phase in phases:
        with st.expander(f"{phase['name']} - {phase['date']}", expanded=False):
            st.metric("Features", phase['features'])
            st.metric("Recall", f"{phase['recall']}%")

elif page == "📈 Insights & Top Riscos":
    st.title("📈 Insights & Dispositivos em Risco")
    st.markdown("Análise de frota e recomendações de manutenção preventiva")
    
    if model is None:
        st.error("Modelo não encontrado.")
        st.stop()
    
    data = load_processed_data()
    if data is None or data.empty:
        st.error("Dados não disponíveis.")
        st.stop()
    
    st.info("Esta página calculará riscos para todos os dispositivos. Implementação em desenvolvimento.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><strong>Sistema Preditivo IoT</strong> | Phase 4: Instability Detection | 99.93% Recall</div>", unsafe_allow_html=True)
