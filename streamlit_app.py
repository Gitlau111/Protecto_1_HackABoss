import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="ETL COVID-19 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones de ETL
@st.cache_data(show_spinner=False)
def importar_datos(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(show_spinner=False)
def transformar(datos):
    df = pd.DataFrame(datos)
    df['date'] = pd.to_datetime(df['date'].astype(str), format="%Y%m%d")
    df = df.sort_values('date').reset_index(drop=True)
    df['daily_cases'] = df['positiveIncrease']
    df['daily_deaths'] = df['deathIncrease']
    return df

# Comentarios para cada análisis
comments = {
    "Gráficos acumulados": "Aquí mostramos la evolución acumulada de muertes e hospitalizaciones para observar tendencias a largo plazo.",
    "Promedios 7 días": "Promedio móvil de 7 días para suavizar la variabilidad diaria. Útil para identificar picos.",
    "Promedios 30 días": "Promedio móvil de 30 días para ver tendencias más estables y comparar con la curva de 7 días.",
    "Serie diarios casos vs muertes": "Comparación directa de casos y muertes diarias para ver retrasos en la mortalidad.",
    "Mapa de calor correlación": "Coeficiente de correlación muestra la relación entre casos diarios y muertes.",
    "Estadísticas descriptivas": "Tabla resumen con medidas como media, mediana y percentiles de las variables.",
    "Distribución antes/después vacunación": "Analizamos cómo cambió el número de casos diarios tras inicio de vacunación (14/12/2020).",
    "Aceleración casos diarios": "Segunda derivada para detectar aceleraciones o desaceleraciones en la transmisión.",
    "Probabilidad hospitalización": "Probabilidad de hospitalización dado un caso positivo (diaria vs acumulada).",
    "Tasa de positividad": "Porcentaje de tests positivos sobre el total de tests realizados cada día."
}

# URL de la API
target_url = "https://api.covidtracking.com/v1/us/daily.json"

# Sidebar: controles\st.sidebar.header("🔧 Parámetros ETL & Visualización")

earliest = datetime(2020, 3, 1)
latest = datetime.today()
fecha_inicio = st.sidebar.date_input(
    "Fecha inicio", value=earliest, min_value=earliest, max_value=latest
)
fecha_fin = st.sidebar.date_input(
    "Fecha fin", value=latest, min_value=earliest, max_value=latest
)

dashboard_ops = list(comments.keys())
seleccion = st.sidebar.multiselect(
    "Elige análisis", dashboard_ops, default=[dashboard_ops[0]]
)

# Carga de datos
if st.sidebar.button("🔄 Cargar datos"):
    raw = importar_datos(target_url)
    df = transformar(raw)
    mask = (
        (df['date'] >= pd.to_datetime(fecha_inicio)) &
        (df['date'] <= pd.to_datetime(fecha_fin))
    )
    df = df.loc[mask].reset_index(drop=True)
    st.session_state['df'] = df
    st.sidebar.success(f"{len(df)} registros cargados")

# Mostrar resultados
if 'df' in st.session_state:
    df = st.session_state['df']
    for op in seleccion:
        st.markdown(f"### {op}")
        # Generar cada gráfico según op
        if op == "Gráficos acumulados":
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(df['date'], df['death'], label="Muertes acumuladas")
            ax.plot(df['date'], df['hospitalizedCumulative'], label="Hospitalizaciones acumuladas")
            ax.legend(); ax.set_xlabel("Fecha"); ax.set_ylabel("Total")
            st.pyplot(fig)
        elif op == "Promedios 7 días":
            df7 = df.resample('7D', on='date')[['daily_cases','daily_deaths']].mean().dropna()
            fig, ax = plt.subplots(figsize=(12,4))
            ax.bar(df7.index - pd.Timedelta(days=3), df7['daily_cases'], width=6, label="Casos 7d")
            ax.bar(df7.index + pd.Timedelta(days=3), df7['daily_deaths'], width=6, label="Muertes 7d")
            ax.legend(); ax.set_xlabel("Periodo"); ax.set_ylabel("Promedio diario")
            st.pyplot(fig)
        elif op == "Promedios 30 días":
            df30 = df.resample('30D', on='date')[['daily_cases','daily_deaths']].mean().dropna()
            fig, ax = plt.subplots(figsize=(12,4))
            ax.bar(df30.index - pd.Timedelta(days=15), df30['daily_cases'], width=20, label="Casos 30d")
            ax.bar(df30.index + pd.Timedelta(days=15), df30['daily_deaths'], width=20, label="Muertes 30d")
            ax.legend(); ax.set_xlabel("Periodo"); ax.set_ylabel("Promedio diario")
            st.pyplot(fig)
        elif op == "Serie diarios casos vs muertes":
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(df['date'], df['daily_cases'], label="Casos diarios")
            ax.plot(df['date'], df['daily_deaths'], label="Muertes diarias")
            ax.legend(); ax.set_xlabel("Fecha"); ax.set_ylabel("Cantidad")
            st.pyplot(fig)
        elif op == "Mapa de calor correlación":
            corr = df[['daily_cases','daily_deaths']].corr()
            fig, ax = plt.subplots(figsize=(4,4))
            sns.heatmap(corr, annot=True, vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
        elif op == "Estadísticas descriptivas":
            st.dataframe(df.describe().T)
        elif op == "Distribución antes/después vacunación":
            fecha_vac = pd.Timestamp("2020-12-14")
            df['period'] = np.where(df['date'] < fecha_vac, 'Antes', 'Después')
            fig, ax = plt.subplots(figsize=(12,4))
            sns.histplot(data=df, x='daily_cases', hue='period', multiple='dodge', bins=30, ax=ax)
            st.pyplot(fig)
        elif op == "Aceleración casos diarios":
            df['acceleration'] = df['daily_cases'].diff().diff()
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(df['date'], df['acceleration'])
            ax.axhline(0, linestyle='--', color='gray')
            st.pyplot(fig)
        elif op == "Probabilidad hospitalización":
            df['prob_hosp_daily'] = df['hospitalizedCurrently'] / df['daily_cases'].replace(0, np.nan)
            df['prob_hosp_acum'] = df['hospitalizedCumulative'] / df['positive'].replace(0, np.nan)
            mean_daily = df['prob_hosp_daily'].mean()
            mean_acum = df['prob_hosp_acum'].mean()
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(['Diaria','Acumulada'], [mean_daily, mean_acum])
            for i,v in enumerate([mean_daily, mean_acum]): ax.text(i, v+0.005, f"{v:.3f}", ha='center')
            st.pyplot(fig)
        elif op == "Tasa de positividad":
            df['positivity_rate'] = df['positive'] / (df['positive']+df['negative'])
            st.write(df['positivity_rate'].describe())
        # Mostrar comentario\        st.markdown(f"**Comentario:** {comments.get(op, '')}")
    # Descarga de datos
    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Descargar CSV", data=csv, file_name="covid_data.csv")
else:
    st.info("📝 Presiona 'Cargar datos' en la barra lateral para iniciar ETL.")
