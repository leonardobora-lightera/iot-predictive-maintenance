import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core import logic as dp
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="EyON - Análise de Bateria",
    page_icon="🔋",
    layout="wide"
)

# Paleta de cores
COLORS = {
    'verde': '#2ECC71',
    'laranja': '#F39C12',
    'vermelho': '#E74C3C',
    'azul': '#3498DB',
    'roxo': '#9B59B6'
}

def get_battery_status(voltage):
    if pd.isna(voltage):
        return 'Desconhecido'
    if voltage >= 3.0:
        return 'OK'
    elif 2.5 <= voltage < 3.0:
        return 'Alerta'
    else:
        return 'Crítico'

def apply_color_styling(val):
    if val == 'OK':
        return f'background-color: {COLORS["verde"]}; color: white'
    elif val == 'Alerta':
        return f'background-color: {COLORS["laranja"]}; color: white'
    elif val == 'Crítico':
        return f'background-color: {COLORS["vermelho"]}; color: white'
    else:
        return ''

def main(df):
    st.title("🔋 Análise de Saúde da Bateria")
    st.markdown("Visão geral da saúde da bateria dos sensores.")
    
    # Exibir os limites de classificação
    st.markdown("""
    ### Limites de Classificação:
    - **OK**: Tensão ≥ 3.0V
    - **Alerta**: 2.5V ≤ Tensão < 3.0V
    - **Crítico**: Tensão < 2.5V
    """)

    if df is not None and not df.empty:
        df = df.copy()

        st.sidebar.header("🔧 Configurações")
        st.sidebar.subheader("Filtros")

        if 'cliente' in df.columns:
            clientes = ['Todos'] + sorted(list(df['cliente'].dropna().unique()))
            selected_cliente = st.sidebar.selectbox("👥 Filtrar por Cliente:", clientes, key='cliente_filter_battery')
        else:
            selected_cliente = 'Todos'

        if 'project_name' in df.columns:
            projetos = ['Todos'] + sorted(list(df['project_name'].dropna().unique()))
            selected_projeto = st.sidebar.selectbox("🏗️ Filtrar por Projeto:", projetos, key='projeto_filter_battery')
        else:
            selected_projeto = 'Todos'

        filtered_df = df.copy()
        if selected_cliente != 'Todos' and 'cliente' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['cliente'] == selected_cliente]
        if selected_projeto != 'Todos' and 'project_name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['project_name'] == selected_projeto]

        if 'battery_voltage' not in filtered_df.columns:
            st.warning("A coluna 'battery_voltage' não foi encontrada nos dados.")
            return

        # Garante que a coluna existe antes de prosseguir
        filtered_df.dropna(subset=['battery_voltage'], inplace=True)
        
        # Converter valores de tensão acima de 5V de mV para V em todo o dataset
        filtered_df['battery_voltage'] = filtered_df['battery_voltage'].apply(
            lambda x: x / 1000 if x > 5 else x
        )

        final_df = filtered_df.sort_values('@timestamp').groupby('device_id').tail(1)

        if not final_df.empty:
            final_df['battery_status'] = final_df['battery_voltage'].apply(get_battery_status)

            # Filtrar dados dos últimos 30 dias
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_data = filtered_df[filtered_df['@timestamp'] >= thirty_days_ago]
            
            # Métricas gerais (apenas sensores únicos dos últimos 30 dias)
            unique_sensors_30days = recent_data['device_id'].nunique()
            latest_status_30days = recent_data.sort_values('@timestamp').groupby('device_id').tail(1)
            latest_status_30days['battery_status'] = latest_status_30days['battery_voltage'].apply(get_battery_status)
            
            bateria_ok = latest_status_30days[latest_status_30days['battery_status'] == 'OK']['device_id'].nunique()
            bateria_alerta = latest_status_30days[latest_status_30days['battery_status'] == 'Alerta']['device_id'].nunique()
            bateria_critico = latest_status_30days[latest_status_30days['battery_status'] == 'Crítico']['device_id'].nunique()

            # Exibir métricas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Sensores", unique_sensors_30days)
            with col2:
                st.metric("Bateria OK", bateria_ok)
            with col3:
                st.metric("Bateria em Alerta", bateria_alerta)
            with col4:
                st.metric("Bateria Crítica", bateria_critico)
            st.markdown("---")

            # Análise de variação de tensão nos últimos 30 dias - Gráfico de Dispersão
            st.subheader("Relação entre Variação de Tensão e Tensão Final")
            
            if not recent_data.empty:
                # Calcular variação de tensão por dispositivo
                voltage_variation = recent_data.groupby('device_id')['battery_voltage'].agg(
                    first_voltage=('first'), 
                    last_voltage=('last')
                )
                voltage_variation['voltage_variation'] = voltage_variation['last_voltage'] - voltage_variation['first_voltage']
                voltage_variation['abs_variation'] = voltage_variation['voltage_variation'].abs()
                
                # Usar todos os sensores, não apenas os de alta variação
                all_sensors = voltage_variation.reset_index()
                
                # Adicionar informações do cliente e projeto
                all_sensors_info = all_sensors.merge(
                    filtered_df[['device_id', 'cliente', 'project_name']].drop_duplicates(),
                    on='device_id',
                    how='left'
                )
                
                # Criar tabela com todos os sensores
                variation_table = all_sensors_info[[
                    'device_id', 'cliente', 'project_name', 
                    'first_voltage', 'last_voltage', 'voltage_variation', 'abs_variation'
                ]].rename(columns={
                    'device_id': 'ID do Sensor',
                    'cliente': 'Cliente',
                    'project_name': 'Projeto',
                    'first_voltage': 'Tensão Inicial (V)',
                    'last_voltage': 'Tensão Final (V)',
                    'voltage_variation': 'Variação (V)',
                    'abs_variation': 'Variação Absoluta (V)'
                })
                
                # Adicionar campo para destacar um sensor específico
                st.sidebar.subheader("Destacar Sensor")
                device_to_highlight = st.sidebar.text_input("Digite o ID do Sensor para destacar:", "")
                
                # Função para determinar a cor de cada ponto com base nas novas regras
                def determine_point_color_and_size(row):
                    variation = row['Variação (V)']
                    final_voltage = row['Tensão Final (V)']
                    abs_variation = abs(variation)
                    
                    # Verde intenso: tensão final > 3.0V e variação entre -0.2V e 0.2V
                    if final_voltage > 3.0 and -0.2 <= variation <= 0.2:
                        return 'green_intense', 15  # Tamanho fixo maior
                    
                    # Vermelho intenso: tensão < 2.5V e |variação| > 0.5V
                    if final_voltage < 2.5 and abs_variation > 0.5:
                        return 'red_intense', 15  # Tamanho fixo maior
                    
                    # Para outros pontos, calcular um valor para o gradiente
                    # Baseado na combinação de tensão final e variação
                    # Normalizar os valores para criar um índice entre 0 e 1
                    # 0 = mais crítico (vermelho), 1 = mais estável (verde)
                    
                    # Normalizar a tensão (2.5V a 3.5V mapeado para 0 a 1)
                    normalized_voltage = min(1, max(0, (final_voltage - 2.5) / 1.0))
                    
                    # Normalizar a variação (0 a 1V mapeado para 0 a 1)
                    # Quanto mais próximo de 0, melhor
                    normalized_variation = max(0, 1 - abs_variation)
                    
                    # Combinar os dois fatores (média ponderada)
                    # Dar mais peso à tensão quando ela está boa, e mais peso à variação quando ela está baixa
                    if final_voltage > 3.0:
                        combined_score = 0.7 * normalized_voltage + 0.3 * normalized_variation
                    else:
                        combined_score = 0.3 * normalized_voltage + 0.7 * normalized_variation
                    
                    # Determinar tamanho baseado na variação absoluta (com tamanho mínimo)
                    size = max(8, abs_variation * 30)
                    
                    return combined_score, size
                
                # Aplicar a função para determinar cores e tamanhos
                color_and_size_results = variation_table.apply(determine_point_color_and_size, axis=1)
                color_values = [result[0] for result in color_and_size_results]
                size_values = [result[1] for result in color_and_size_results]
                
                # Converter para DataFrame para facilitar manipulação
                results_df = pd.DataFrame({
                    'color': color_values,
                    'size': size_values
                })
                
                # Separar pontos por categoria
                green_intense_mask = results_df['color'] == 'green_intense'
                red_intense_mask = results_df['color'] == 'red_intense'
                gradient_mask = ~green_intense_mask & ~red_intense_mask
                
                # Criar figura com três traces: verde intenso, vermelho intenso e gradiente
                fig_scatter = go.Figure()
                
                # Adicionar pontos de verde intenso
                green_intense_points = variation_table[green_intense_mask]
                green_intense_sizes = results_df[green_intense_mask]['size']
                if not green_intense_points.empty:
                    fig_scatter.add_trace(go.Scatter(
                        x=green_intense_points['Variação (V)'],
                        y=green_intense_points['Tensão Final (V)'],
                        mode='markers',
                        marker=dict(
                            color='darkgreen',
                            size=green_intense_sizes,
                            line=dict(width=1, color='black')
                        ),
                        name='Estável',
                        hovertemplate='<b>ID do Sensor</b>: %{customdata[0]}<br>' +
                                      '<b>Cliente</b>: %{customdata[1]}<br>' +
                                      '<b>Projeto</b>: %{customdata[2]}<br>' +
                                      '<b>Tensão Inicial</b>: %{customdata[3]:.2f}V<br>' +
                                      '<b>Tensão Final</b>: %{customdata[4]:.2f}V<br>' +
                                      '<b>Variação</b>: %{customdata[5]:.2f}V<extra></extra>',
                        customdata=green_intense_points[['ID do Sensor', 'Cliente', 'Projeto', 
                                                       'Tensão Inicial (V)', 'Tensão Final (V)', 'Variação (V)']].values
                    ))
                
                # Adicionar pontos de vermelho intenso
                red_intense_points = variation_table[red_intense_mask]
                red_intense_sizes = results_df[red_intense_mask]['size']
                if not red_intense_points.empty:
                    fig_scatter.add_trace(go.Scatter(
                        x=red_intense_points['Variação (V)'],
                        y=red_intense_points['Tensão Final (V)'],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=red_intense_sizes,
                            line=dict(width=1, color='darkred')
                        ),
                        name='Crítico',
                        hovertemplate='<b>ID do Sensor</b>: %{customdata[0]}<br>' +
                                      '<b>Cliente</b>: %{customdata[1]}<br>' +
                                      '<b>Projeto</b>: %{customdata[2]}<br>' +
                                      '<b>Tensão Inicial</b>: %{customdata[3]:.2f}V<br>' +
                                      '<b>Tensão Final</b>: %{customdata[4]:.2f}V<br>' +
                                      '<b>Variação</b>: %{customdata[5]:.2f}V<extra></extra>',
                        customdata=red_intense_points[['ID do Sensor', 'Cliente', 'Projeto', 
                                                     'Tensão Inicial (V)', 'Tensão Final (V)', 'Variação (V)']].values
                    ))
                
                # Adicionar pontos com gradiente
                gradient_points = variation_table[gradient_mask]
                gradient_colors = results_df[gradient_mask]['color']
                gradient_sizes = results_df[gradient_mask]['size']
                if not gradient_points.empty:
                    fig_scatter.add_trace(go.Scatter(
                        x=gradient_points['Variação (V)'],
                        y=gradient_points['Tensão Final (V)'],
                        mode='markers',
                        marker=dict(
                            color=gradient_colors,
                            colorscale=['red', 'yellow', 'green'],
                            size=gradient_sizes,
                            colorbar=dict(title="Estabilidade"),
                            line=dict(width=1, color='black')
                        ),
                        name='Intermediário',
                        hovertemplate='<b>ID do Sensor</b>: %{customdata[0]}<br>' +
                                      '<b>Cliente</b>: %{customdata[1]}<br>' +
                                      '<b>Projeto</b>: %{customdata[2]}<br>' +
                                      '<b>Tensão Inicial</b>: %{customdata[3]:.2f}V<br>' +
                                      '<b>Tensão Final</b>: %{customdata[4]:.2f}V<br>' +
                                      '<b>Variação</b>: %{customdata[5]:.2f}V<extra></extra>',
                        customdata=gradient_points[['ID do Sensor', 'Cliente', 'Projeto', 
                                                  'Tensão Inicial (V)', 'Tensão Final (V)', 'Variação (V)']].values
                    ))
                
                # Destacar sensor específico, se fornecido
                if device_to_highlight:
                    # Converter device_to_highlight para o mesmo tipo da coluna 'ID do Sensor'
                    # e verificar se está presente nos dados
                    device_to_highlight_clean = str(device_to_highlight).strip()
                    matching_sensors = variation_table[variation_table['ID do Sensor'].astype(str).str.strip() == device_to_highlight_clean]
                    
                    if not matching_sensors.empty:
                        highlighted_sensor = matching_sensors.iloc[0]  # Pegar a primeira ocorrência se houver mais de uma
                        # Certificar-se de que os valores são escalares
                        voltage_variation_val = float(highlighted_sensor['Variação (V)'])
                        final_voltage_val = float(highlighted_sensor['Tensão Final (V)'])
                        
                        fig_scatter.add_trace(go.Scatter(
                            x=[voltage_variation_val],
                            y=[final_voltage_val],
                            mode='markers+text',
                            marker=dict(
                                color='black',
                                size=20,
                                symbol='circle-open',
                                line=dict(width=4, color='black')
                            ),
                            text=[device_to_highlight_clean],
                            textposition="top center",
                            textfont=dict(size=14, color="black"),
                            name='Sensor Destacado',
                            showlegend=False,
                            hovertemplate='<b>ID do Sensor</b>: %{text}<br>' +
                                          '<b>Tensão Final</b>: %{y:.2f}V<br>' +
                                          '<b>Variação</b>: %{x:.2f}V<extra></extra>',
                        ))
                        st.info(f"Sensor {device_to_highlight_clean} destacado no gráfico com um círculo preto.")
                    else:
                        st.warning(f"Sensor {device_to_highlight_clean} não encontrado nos dados.")
                        # Mostrar alguns IDs disponíveis para facilitar a identificação
                        available_ids = variation_table['ID do Sensor'].astype(str).str.strip().head(5).tolist()
                        st.info(f"Alguns IDs disponíveis: {', '.join(available_ids)}")
                fig_scatter.update_layout(
                    title='Relação entre Variação de Tensão e Tensão Final (Últimos 30 Dias)',
                    xaxis_title="Variação de Tensão (V)",
                    yaxis_title="Tensão Final (V)",
                    legend_title="Classificação",
                    showlegend=True
                )
                # Adicionar linha vertical no zero para facilitar a visualização
                fig_scatter.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
                # Adicionar linhas horizontais para os limites de classificação
                fig_scatter.add_hline(y=3.0, line_dash="dot", line_color=COLORS['verde'], 
                                      annotation_text="Limite OK", annotation_position="top left")
                fig_scatter.add_hline(y=2.5, line_dash="dot", line_color=COLORS['laranja'], 
                                      annotation_text="Limite Crítico", annotation_position="bottom right")
                
                # Adicionar anotações explicativas
                fig_scatter.add_annotation(
                    x=0, y=3.2, text="Estável", showarrow=False, 
                    bgcolor="darkgreen", font=dict(color="white")
                )
                fig_scatter.add_annotation(
                    x=0.7, y=2.3, text="Crítico", showarrow=False, 
                    bgcolor="red", font=dict(color="white")
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("ℹ️ Não há dados suficientes dos últimos 30 dias para análise de variação.")
            
            st.markdown("---")

            st.subheader("Distribuição do Nível de Bateria")
            status_counts = final_df['battery_status'].value_counts().reset_index()
            status_counts.columns = ['status', 'count']

            fig_bar = px.bar(status_counts, x='status', y='count', title='Distribuição dos Status de Bateria',
                               labels={'status': 'Status da Bateria', 'count': 'Número de Sensores'},
                               color='status',
                               color_discrete_map={'OK': COLORS['verde'], 'Alerta': COLORS['laranja'], 'Crítico': COLORS['vermelho']})
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(status_counts, names='status', values='count', title='Distribuição Percentual dos Status de Bateria',
                             color='status',
                             color_discrete_map={'OK': COLORS['verde'], 'Alerta': COLORS['laranja'], 'Crítico': COLORS['vermelho']})
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Histórico de Tensão da Bateria")
            
            # Filtrar dados dos últimos 30 dias
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_data = filtered_df[filtered_df['@timestamp'] >= thirty_days_ago]
            
            # Converter valores de tensão acima de 5V de mV para V
            recent_data['battery_voltage'] = recent_data['battery_voltage'].apply(
                lambda x: x / 1000 if x > 5 else x
            )
            
            # Verificar se há dados antes de criar o seletor
            if not recent_data.empty:
                # Criar um seletor para escolher um sensor específico (apenas sensores com dados nos últimos 30 dias)
                device_ids = sorted(recent_data['device_id'].unique())
                selected_device = st.selectbox("Selecione um Sensor (Device ID):", device_ids)
                
                # Filtrar os dados para o sensor selecionado
                sensor_data = recent_data[recent_data['device_id'] == selected_device]
                
                if not sensor_data.empty:
                    # Criar o gráfico para o sensor selecionado
                    fig_history = px.line(sensor_data, x='@timestamp', y='battery_voltage',
                                          title=f'Evolução da Tensão da Bateria - Sensor {selected_device}',
                                          labels={'@timestamp': 'Data', 'battery_voltage': 'Tensão (V)'})
                    fig_history.update_layout(showlegend=False)
                    fig_history.add_hline(y=3.0, line_dash="dash", line_color=COLORS['verde'], 
                                          annotation_text="Limite OK", annotation_position="top left")
                    fig_history.add_hline(y=2.5, line_dash="dash", line_color=COLORS['laranja'], 
                                          annotation_text="Limite Crítico", annotation_position="bottom right")
                    st.plotly_chart(fig_history, use_container_width=True)
                else:
                    st.info("Não há dados de bateria para o sensor selecionado.")
            else:
                st.info("Não há dados de bateria para exibir no período selecionado.")


            # Tabela de Resumo de Variação de Tensão (Últimos 30 Dias)
            st.markdown("---")
            st.subheader("Resumo de Variação de Tensão (Últimos 30 Dias)")
            
            # Filtrar dados dos últimos 30 dias
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_data = filtered_df[filtered_df['@timestamp'] >= thirty_days_ago]
            
            if not recent_data.empty:
                # Calcular variação de tensão por dispositivo
                voltage_variation = recent_data.groupby('device_id')['battery_voltage'].agg(
                    first_voltage=('first'), 
                    last_voltage=('last')
                )
                voltage_variation['voltage_variation'] = voltage_variation['last_voltage'] - voltage_variation['first_voltage']
                voltage_variation['abs_variation'] = voltage_variation['voltage_variation'].abs()
                
                # Usar todos os sensores, não apenas os de alta variação
                all_sensors = voltage_variation.reset_index()
                
                # Adicionar informações do cliente e projeto
                all_sensors_info = all_sensors.merge(
                    filtered_df[['device_id', 'cliente', 'project_name']].drop_duplicates(),
                    on='device_id',
                    how='left'
                )
                
                # Adicionar a tensão da bateria mais recente para cada sensor
                latest_battery_info = recent_data.sort_values('@timestamp').groupby('device_id').tail(1)[['device_id', 'battery_voltage']]
                all_sensors_info = all_sensors_info.merge(latest_battery_info, on='device_id', how='left')
                
                # Criar tabela com todos os sensores (adicionando battery_voltage)
                variation_table = all_sensors_info[[
                    'device_id', 'cliente', 'project_name', 'battery_voltage',
                    'first_voltage', 'last_voltage', 'voltage_variation', 'abs_variation'
                ]].rename(columns={
                    'device_id': 'ID do Sensor',
                    'cliente': 'Cliente',
                    'project_name': 'Projeto',
                    'battery_voltage': 'Tensão da Bateria (V)',
                    'first_voltage': 'Tensão Inicial (V)',
                    'last_voltage': 'Tensão Final (V)',
                    'voltage_variation': 'Variação (V)',
                    'abs_variation': 'Variação Absoluta (V)'
                })
                
                # Aplicar estilo condicional para variações positivas e negativas
                def variation_color(val):
                    if val > 0:
                        return 'color: green'
                    elif val < 0:
                        return 'color: red'
                    else:
                        return ''
                
                styled_variation = variation_table.style.applymap(
                    variation_color, 
                    subset=['Variação (V)']
                )
                st.dataframe(styled_variation, use_container_width=True)
            else:
                st.info("ℹ️ Não há dados suficientes dos últimos 30 dias para análise de variação.")

        else:
            st.warning("Não foram encontrados dados de bateria para os filtros selecionados.")

if __name__ == "__main__":
    main()