import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Configuração da página do Streamlit
st.set_page_config(page_title="Dashboard de Produtividade Acadêmica", layout="wide")

# Desabilitando o limite de linhas do Altair
alt.data_transformers.disable_max_rows()

# =====================================================================
# CARREGAMENTO E PROCESSAMENTO DE DADOS (Com Cache para Performance)
# =====================================================================
@st.cache_data
def load_and_process_data():
    # 1. Carregando os dados originais
    # Substitua pelo caminho correto caso não esteja na mesma pasta
    df = pd.read_csv('student_productivity_distraction_dataset_20000.csv')
    
    # Adicionando as colunas calculadas que você usou na Etapa 2.4
    df['total_distraction_hours'] = (
        df['phone_usage_hours'] + df['social_media_hours'] + 
        df['youtube_hours'] + df['gaming_hours']
    )
    df['distraction_bins'] = pd.cut(
        df['total_distraction_hours'], 
        bins=[0, 5, 10, 15, 20, np.inf],
        labels=['0-5h', '5-10h', '10-15h', '15-20h', '20h+']
    )
    df['productivity_quartile'] = pd.qcut(
        df['productivity_score'], q=4,
        labels=['1º Quartil (Menos Produtivos)', '2º Quartil', '3º Quartil', '4º Quartil (Mais Produtivos)']
    )

    # 2. Gerando Amostras
    df_pop = df.copy()
    df_pop['Método'] = '01. População (20000)'
    
    df_simples = df.sample(n=1000, random_state=42).copy()
    df_simples['Método'] = '02. Aleatória Simples'
    
    df_estratificada = df.groupby('gender', group_keys=False).apply(lambda x: x.sample(n=333, random_state=42, replace=True)).copy()
    df_estratificada['Método'] = '03. Estratificada (50/50 Gênero)'
    
    df_sistematica = df.iloc[::20].copy()
    df_sistematica['Método'] = '04. Sistemática (Passo 10)'
    
    df_pps = df.sample(n=1000, weights='productivity_score', random_state=42).copy()
    df_pps['Método'] = '07. PPS (Peso: Productivity)'
    
    df_conveniencia = df.head(1000).copy()
    df_conveniencia['Método'] = '08. Conveniência (Head)'

    df_comparacao = pd.concat([df_pop, df_simples, df_estratificada, df_sistematica, df_pps, df_conveniencia])

    # 3. Avaliação de Métodos (TVD e MAPE)
    pop_productivity_mean = df_pop['productivity_score'].mean()
    pop_attendance_mean = df_pop['attendance_percentage'].mean()
    pop_cat_prop = df_pop['age'].value_counts(normalize=True)
    pop_gender_prop = df_pop['gender'].value_counts(normalize=True)

    resultados = []
    for metodo in df_comparacao['Método'].unique():
        if 'População' in metodo: continue
        
        df_m = df_comparacao[df_comparacao['Método'] == metodo]
        err_productivity = abs(df_m['productivity_score'].mean() - pop_productivity_mean) / pop_productivity_mean
        err_attendance = abs(df_m['attendance_percentage'].mean() - pop_attendance_mean) / pop_attendance_mean
        
        m_cat_prop = df_m['age'].value_counts(normalize=True).reindex(pop_cat_prop.index, fill_value=0)
        tvd_cat = 0.5 * np.sum(np.abs(pop_cat_prop - m_cat_prop))
        
        m_gender_prop = df_m['gender'].value_counts(normalize=True).reindex(pop_gender_prop.index, fill_value=0)
        tvd_gender = 0.5 * np.sum(np.abs(pop_gender_prop - m_gender_prop))
        
        divergencia_total = err_productivity + err_attendance + tvd_cat + tvd_gender
        resultados.append({
            'Método': metodo,
            'Divergência Total': divergencia_total
        })

    df_ranking = pd.DataFrame(resultados).sort_values('Divergência Total').reset_index(drop=True)
    metodo_vencedor = df_ranking.iloc[0]['Método']
    df_amostra = df_comparacao[df_comparacao['Método'] == metodo_vencedor].drop(columns=['Método']).copy()

    return df, df_amostra, df_ranking

# Carrega os dados na memória do Streamlit
df_original, df_amostra, df_ranking = load_and_process_data()

# =====================================================================
# INTERFACE DO STREAMLIT
# =====================================================================
st.title("📊 Dashboard: Produtividade e Distrações de Estudantes")
st.markdown("Análise baseada em técnicas de amostragem e visualizações interativas com Altair.")

# Criando abas para organizar o dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛠️ Amostragem e Dados", 
    "🏃 Hábitos Diários", 
    "📱 Distrações", 
    "🎓 Desempenho Acadêmico", 
    "🔗 Correlações"
])

# --- ABA 1: AMOSTRAGEM ---
with tab1:
    st.header("Metodologia de Amostragem")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Tamanho da População", f"{df_original.shape[0]} linhas")
        st.metric("Tamanho da Amostra Selecionada", f"{df_amostra.shape[0]} linhas")
        
    with col2:
        st.write("🏆 **Ranking dos Métodos de Amostragem (Menor Divergência Total = Melhor)**")
        st.dataframe(df_ranking.style.background_gradient(subset=['Divergência Total'], cmap='RdYlGn_r'), use_container_width=True)
        st.success(f"Método vencedor selecionado automaticamente: **{df_ranking.iloc[0]['Método']}**")
    
    st.subheader("Visualização dos Dados Originais")
    st.dataframe(df_original.head(), use_container_width=True)

# --- ABA 2: HÁBITOS DIÁRIOS ---
with tab2:
    st.header("Raio-X dos Hábitos Diários")
    st.markdown("Analisando o impacto isolado do sono, exercício físico, pausas e ingestão de café na produtividade.")
    
    LARGURA, ALTURA = 220, 300
    
    def criar_grafico_habito(x_col, x_title, cor):
        pontos = alt.Chart(df_amostra).mark_circle(opacity=0.4, color=cor, size=40).encode(
            x=alt.X(f'{x_col}:Q', title=x_title, scale=alt.Scale(zero=False)),
            y=alt.Y('productivity_score:Q', title='Score Produtividade', scale=alt.Scale(zero=False)),
            tooltip=[x_col, 'productivity_score']
        )
        tendencia = pontos.transform_regression(x_col, 'productivity_score').mark_line(color='#c0392b', size=3)
        return (pontos + tendencia).properties(width=LARGURA, height=ALTURA)

    g_sono = criar_grafico_habito('sleep_hours', 'Horas de Sono', '#3498db')
    g_exercicio = criar_grafico_habito('exercise_minutes', 'Minutos de Exercício', '#2ecc71')
    g_pausas = criar_grafico_habito('breaks_per_day', 'Pausas por Dia', '#9b59b6')
    g_cafe = criar_grafico_habito('coffee_intake_mg', 'Café (mg)', '#e67e22')

    painel_habitos = (g_sono | g_exercicio | g_pausas | g_cafe).configure_axis(labelFontSize=12, titleFontSize=14)
    st.altair_chart(painel_habitos, use_container_width=True)

# --- ABA 3: DISTRAÇÕES ---
with tab3:
    st.header("Análise de Foco e Distrações")
    
    # Parte 1: Interatividade Foco vs Distração
    distraction_vars = ['phone_usage_hours', 'social_media_hours', 'youtube_hours', 'gaming_hours']
    df_distractions = df_amostra.melt(
        id_vars=['student_id', 'focus_score', 'productivity_score'],
        value_vars=distraction_vars, var_name='tipo_distracao', value_name='horas_gastas'
    )

    distraction_options = df_distractions['tipo_distracao'].unique().tolist()
    dropdown = alt.binding_select(options=distraction_options, name='Escolha a Distração: ')
    distraction_select = alt.selection_point(fields=['tipo_distracao'], bind=dropdown, value=distraction_options[0])
    brush = alt.selection_interval(encodings=['x'])

    hist_focus = alt.Chart(df_amostra).mark_bar().encode(
        x=alt.X('focus_score:Q', bin=True, title='Distribuição do Score de Foco'),
        y=alt.Y('count()', title='Número de Estudantes'),
        color=alt.condition(brush, alt.value('#2c3e50'), alt.value('lightgray'))
    ).properties(height=150, title='Selecione uma faixa de Foco arrastando o mouse').add_params(brush)

    points = alt.Chart(df_distractions).mark_point(opacity=0.3, size=40, filled=True).encode(
        x=alt.X('horas_gastas:Q', title='Horas Gastas por Dia'),
        y=alt.Y('productivity_score:Q', title='Score de Produtividade'),
        color=alt.value('#606060')
    )

    trend_line = points.transform_regression('horas_gastas', 'productivity_score').mark_line(color='#c0392b', size=4)
    
    main_distraction_plot = alt.layer(points, trend_line).add_params(distraction_select).transform_filter(distraction_select).transform_filter(brush).properties(height=350)
    
    st.altair_chart(hist_focus & main_distraction_plot, use_container_width=True)
    st.divider()
    
    # Parte 2: Boxplot e Barras Empilhadas
    ordem_faixas = ['0-5h', '5-10h', '10-15h', '15-20h', '20h+']
    boxplot = alt.Chart(df_original).mark_boxplot(extent='min-max', size=40).encode(
        x=alt.X('distraction_bins:O', title='Total de Distração (Horas/Dia)', sort=ordem_faixas),
        y=alt.Y('productivity_score:Q', title='Score de Produtividade', scale=alt.Scale(zero=False)),
        color=alt.Color('distraction_bins:O', legend=None, scale=alt.Scale(scheme='teals'), sort=ordem_faixas)
    ).properties(width=350, height=350, title='Volume de Distração vs Produtividade')

    df_melted = df_original.melt(
        id_vars=['student_id', 'productivity_quartile'],
        value_vars=['phone_usage_hours', 'social_media_hours', 'youtube_hours', 'gaming_hours'],
        var_name='distraction_type', value_name='hours'
    )
    df_melted['distraction_type'] = df_melted['distraction_type'].replace({
        'phone_usage_hours': 'Celular', 'social_media_hours': 'Redes Sociais', 
        'youtube_hours': 'YouTube', 'gaming_hours': 'Jogos'
    })

    stacked_bar = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('productivity_quartile:O', title='Perfil dos Alunos (Quartis)'),
        y=alt.Y('mean(hours):Q', title='Média de Horas Diárias'),
        color=alt.Color('distraction_type:N', title='Tipo de Distração', scale=alt.Scale(scheme='tableau10')),
        tooltip=['productivity_quartile', 'distraction_type', 'mean(hours)']
    ).properties(width=350, height=350, title='Do que os alunos se distraem?')

    st.altair_chart(boxplot | stacked_bar, use_container_width=True)

# --- ABA 4: DESEMPENHO ACADÊMICO ---
with tab4:
    st.header("Desempenho Acadêmico e Produtividade")
    
    st.subheader("1. Produtividade vs Nota Final")
    pontos_choque = alt.Chart(df_amostra).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('productivity_score:Q', title='Score de Produtividade', scale=alt.Scale(zero=False)),
        y=alt.Y('final_grade:Q', title='Nota Final', scale=alt.Scale(zero=False)),
        color=alt.value('#e74c3c'),
        tooltip=['productivity_score', 'final_grade']
    )
    linha_tendencia = pontos_choque.transform_regression('productivity_score', 'final_grade').mark_line(color='#2c3e50', size=4, strokeDash=[5, 5])
    st.altair_chart((pontos_choque + linha_tendencia).properties(height=350), use_container_width=True)
    
    st.divider()
    
    st.subheader("2. Perfil Acadêmico Dinâmico e Produtividade")
    brush2 = alt.selection_interval()
    heatmap_perfil = alt.Chart(df_amostra).mark_rect().encode(
        x=alt.X('focus_score:Q', bin=alt.Bin(maxbins=10), title='Score de Foco'),
        y=alt.Y('attendance_percentage:Q', bin=alt.Bin(maxbins=10), title='Presença %'),
        color=alt.Color('mean(productivity_score):Q', scale=alt.Scale(scheme='turbo'), title='Média Produtividade'),
        tooltip=['mean(focus_score)', 'mean(attendance_percentage)', 'mean(productivity_score)', 'count()']
    ).add_params(brush2).properties(width=400, height=350)

    boxplot_entregas = alt.Chart(df_amostra).mark_boxplot(extent='min-max', size=30).encode(
        x=alt.X('assignments_completed:Q', bin=alt.Bin(maxbins=4), title='Trabalhos Entregues'),
        y=alt.Y('productivity_score:Q', title='Score de Produtividade', scale=alt.Scale(zero=False)),
        color=alt.Color('assignments_completed:Q', bin=alt.Bin(maxbins=4), legend=None, scale=alt.Scale(scheme='teals'))
    ).transform_filter(brush2).properties(width=350, height=350)
    
    st.altair_chart((heatmap_perfil | boxplot_entregas).resolve_scale(color='independent'), use_container_width=True)

    st.divider()

    st.subheader("3. Horas de Estudo, Estresse e Produtividade")
    brush3 = alt.selection_interval(encodings=['x', 'y'])
    scatter_estudo = alt.Chart(df_original).mark_circle(opacity=0.4, size=30).encode(
        x=alt.X('study_hours_per_day:Q', title='Horas de Estudo'),
        y=alt.Y('productivity_score:Q', title='Score de Produtividade', scale=alt.Scale(zero=False)),
        color=alt.condition(brush3, alt.value('#2ca02c'), alt.value('lightgray')),
        tooltip=['study_hours_per_day', 'productivity_score', 'stress_level']
    ).add_params(brush3).properties(width=400, height=350)

    bar_estresse = alt.Chart(df_original).mark_bar().encode(
        x=alt.X('stress_level:O', title='Nível de Estresse (0 a 10)'),
        y=alt.Y('mean(productivity_score):Q', title='Produtividade Média'),
        color=alt.Color('stress_level:Q', scale=alt.Scale(scheme='oranges'), legend=None)
    ).transform_filter(brush3).properties(width=400, height=350)

    st.altair_chart(scatter_estudo | bar_estresse, use_container_width=True)

# --- ABA 5: CORRELAÇÕES ---
with tab5:
    st.header("Correlação de Pearson")
    
    df_numerico = df_amostra.select_dtypes(include=[np.number])
    df_correlacao = df_numerico.corr(method='pearson').stack().reset_index()
    df_correlacao.columns = ['var_1', 'var_2', 'correlacao']

    matriz_correlacao = alt.Chart(df_correlacao).mark_rect().encode(
        x=alt.X('var_1:N', title=None),
        y=alt.Y('var_2:N', title=None),
        color=alt.Color('correlacao:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1]), title='Correlação'),
        tooltip=['var_1', 'var_2', alt.Tooltip('correlacao', format='.3f')]
    ).properties(height=600)

    st.altair_chart(matriz_correlacao, use_container_width=True)
