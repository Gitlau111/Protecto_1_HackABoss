import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Configuración de la página
st.set_page_config(
    page_title="ETL COVID-19 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar clave API de OpenAI desde secretos
oai_key = st.secrets.get("openai_api_key", None)
if not oai_key:
    st.warning("🔑 No se encontró la clave de OpenAI. Agrega 'openai_api_key' a Streamlit secrets.")
else:
    openai.api_key = oai_key

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

# Sidebar: Controles ETL y Visualización
st.sidebar.header("🔧 Parámetros ETL & Visualización")

earliest = datetime(2020, 3, 1)
latest = datetime.today()
fecha_inicio = st.sidebar.date_input("Fecha inicio", value=earliest, min_value=earliest, max_value=latest)
fecha_fin = st.sidebar.date_input("Fecha fin", value=latest,    min_value=earliest, max_value=latest)

dashboard_ops = list(comments.keys())
seleccion = st.sidebar.multiselect("Elige análisis", dashboard_ops, default=[dashboard_ops[0]])

# Chatbot en sidebar\st.sidebar.header("💬 Chat COVID-19")
if oai_key:
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    user_input = st.sidebar.chat_input("Pregunta sobre COVID-19 ETL")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        bot_msg = response.choices[0].message
        st.session_state.messages.append({"role": bot_msg.role, "content": bot_msg.content})
    for msg in st.session_state.messages:
        st.sidebar.chat_message(msg['role'], msg['content'])
else:
    st.sidebar.info("🔒 Chat desactivado: falta API key.")

# Botón de carga de datos
if st.sidebar.button("🔄 Cargar datos"):
    raw = importar_datos(target_url)
    df = transformar(raw)
    mask = (df['date'] >= pd.to_datetime(fecha_inicio)) & (df['date'] <= pd.to_datetime(fecha_fin))
    df = df.loc[mask].reset_index(drop=True)
    st.session_state['df'] = df
    st.sidebar.success(f"{len(df)} registros cargados.")

# Main: mostrar análisis
st.title("📊 Dashboard ETL COVID-19 (EEUU)")
if 'df' in st.session_state:
    df = st.session_state['df']
    for op in seleccion:
        st.markdown(f"### {op}")
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
        # ... restantes análisis como antes ...
        # Finalmente mostrar comentario
        st.markdown(f"**Comentario:** {comments.get(op, '')}")
    # Descarga de datos
    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Descargar CSV", data=csv, file_name="covid_data.csv")
else:
    st.info("📝 Presiona 'Cargar datos' en la barra lateral para iniciar ETL.")
