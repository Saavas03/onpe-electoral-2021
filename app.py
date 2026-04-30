import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN ──
st.set_page_config(
    page_title="ONPE 2021 - Segunda Vuelta",
    page_icon="🗳️",
    layout="wide"
)

# ── ESTILOS ──
st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #c0392b, #8e1a0e);
        padding: 25px; border-radius: 12px;
        text-align: center; color: white; margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# ── ENCABEZADO ──
st.markdown("""
<div class="header">
    <h1>🗳️ OFICINA NACIONAL DE PROCESOS ELECTORALES — ONPE</h1>
    <h2>Resultados Electorales Segunda Vuelta 2021</h2>
    <p>Pedro Castillo (Perú Libre) vs Keiko Fujimori (Fuerza Popular)</p>
    <p>Fecha: 6 de junio de 2021 | Fuente: datosabiertos.gob.pe</p>
</div>
""", unsafe_allow_html=True)

# ── CARGA DE DATOS ──
@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        "data/resultados_2da_vuelta.csv",
        encoding="latin-1",
        sep=";"
    )
    # Limpiar espacios en nombres de columnas
    df.columns = df.columns.str.strip()

    # Limpiar comillas en columnas de texto
    for col in ["DEPARTAMENTO", "PROVINCIA", "DISTRITO"]:
        if col in df.columns:
            df[col] = df[col].str.replace('"', '').str.strip()

    # Convertir votos a numérico
    cols_votos = ["VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN", "VOTOS_VI", "N_ELEC_HABIL"]
    for col in cols_votos:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Calcular columnas derivadas
    df["VOTOS_VALIDOS"] = df["VOTOS_P1"] + df["VOTOS_P2"]
    df["VOTOS_TOTAL"]   = df["VOTOS_P1"] + df["VOTOS_P2"] + df["VOTOS_VB"] + df["VOTOS_VN"] + df["VOTOS_VI"]
    df["PCT_CASTILLO"]  = (df["VOTOS_P1"] / df["VOTOS_VALIDOS"].replace(0, np.nan) * 100).round(2)
    df["PCT_FUJIMORI"]  = (df["VOTOS_P2"] / df["VOTOS_VALIDOS"].replace(0, np.nan) * 100).round(2)
    df["GANADOR_MESA"]  = np.where(
        df["VOTOS_P1"] > df["VOTOS_P2"],
        "Pedro Castillo", "Keiko Fujimori"
    )
    return df

with st.spinner("⏳ Cargando datos oficiales ONPE..."):
    df = cargar_datos()

st.success(f"✅ Dataset cargado correctamente: {len(df):,} mesas de sufragio")

# ── SIDEBAR — FILTROS ──
st.sidebar.title("🔍 Filtros de consulta")
st.sidebar.markdown("---")

departamentos = ["TODOS"] + sorted(df["DEPARTAMENTO"].dropna().unique().tolist())
depa_sel = st.sidebar.selectbox("📍 Departamento:", departamentos)

if depa_sel != "TODOS":
    df_filtrado = df[df["DEPARTAMENTO"] == depa_sel]
else:
    df_filtrado = df.copy()

st.sidebar.markdown("---")
st.sidebar.metric("Mesas mostradas", f"{len(df_filtrado):,}")
st.sidebar.markdown("**Fuente:** ONPE — datosabiertos.gob.pe")

# ── PESTAÑAS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen Nacional",
    "🗺️ Por Región",
    "📈 Visualizaciones",
    "🤖 Machine Learning",
    "📋 Datos por Mesa"
])

# ─────────────────────────────────
# TAB 1 — RESUMEN NACIONAL
# ─────────────────────────────────
with tab1:
    st.header("📊 Resumen Nacional — Segunda Vuelta 2021")

    total_mesas    = len(df)
    total_castillo = int(df["VOTOS_P1"].sum())
    total_fujimori = int(df["VOTOS_P2"].sum())
    total_validos  = int(df["VOTOS_VALIDOS"].sum())
    total_nulos    = int(df["VOTOS_VN"].sum())
    total_blancos  = int(df["VOTOS_VB"].sum())
    total_impug    = int(df["VOTOS_VI"].sum())

    pct_cast = total_castillo / total_validos * 100
    pct_fuji = total_fujimori / total_validos * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🗳️ Total Mesas",       f"{total_mesas:,}")
    col2.metric("✅ Votos Válidos",      f"{total_validos:,}")
    col3.metric("❌ Votos Nulos",        f"{total_nulos:,}")
    col4.metric("⬜ Votos en Blanco",   f"{total_blancos:,}")

    st.markdown("---")
    col5, col6 = st.columns(2)
    col5.markdown(f"""
    ### 🔴 Pedro Castillo — Perú Libre
    - **Votos:** {total_castillo:,}
    - **Porcentaje:** {pct_cast:.3f}%
    - **Mesas ganadas:** {(df['GANADOR_MESA'] == 'Pedro Castillo').sum():,}
    """)
    col6.markdown(f"""
    ### ⚫ Keiko Fujimori — Fuerza Popular
    - **Votos:** {total_fujimori:,}
    - **Porcentaje:** {pct_fuji:.3f}%
    - **Mesas ganadas:** {(df['GANADOR_MESA'] == 'Keiko Fujimori').sum():,}
    """)

    # Gráfico de barras
    fig_bar = go.Figure([
        go.Bar(name="Pedro Castillo", x=["Pedro Castillo"],
               y=[total_castillo], marker_color="#c0392b",
               text=[f"{pct_cast:.3f}%"], textposition="outside"),
        go.Bar(name="Keiko Fujimori", x=["Keiko Fujimori"],
               y=[total_fujimori], marker_color="#2c3e50",
               text=[f"{pct_fuji:.3f}%"], textposition="outside")
    ])
    fig_bar.update_layout(
        title="Votos Totales por Candidato — Nivel Nacional",
        yaxis_title="Número de votos", height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Pie chart
    fig_pie = px.pie(
        values=[total_castillo, total_fujimori, total_blancos, total_nulos, total_impug],
        names=["Castillo", "Fujimori", "Blancos", "Nulos", "Impugnados"],
        title="Distribución de todos los votos emitidos",
        color_discrete_sequence=["#c0392b","#2c3e50","#bdc3c7","#7f8c8d","#95a5a6"]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────
# TAB 2 — POR REGIÓN
# ─────────────────────────────────
with tab2:
    st.header("🗺️ Resultados por Departamento")

    por_depa = df.groupby("DEPARTAMENTO").agg(
        VOTOS_CASTILLO=("VOTOS_P1", "sum"),
        VOTOS_FUJIMORI=("VOTOS_P2", "sum"),
        VOTOS_VALIDOS =("VOTOS_VALIDOS", "sum"),
        VOTOS_NULOS   =("VOTOS_VN", "sum"),
        TOTAL_MESAS   =("MESA_DE_VOTACION", "count")
    ).reset_index()

    por_depa["PCT_CASTILLO"] = (por_depa["VOTOS_CASTILLO"] / por_depa["VOTOS_VALIDOS"] * 100).round(2)
    por_depa["PCT_FUJIMORI"] = (por_depa["VOTOS_FUJIMORI"] / por_depa["VOTOS_VALIDOS"] * 100).round(2)
    por_depa["GANADOR"]      = np.where(
        por_depa["VOTOS_CASTILLO"] > por_depa["VOTOS_FUJIMORI"],
        "Pedro Castillo", "Keiko Fujimori"
    )

    fig_reg = px.bar(
        por_depa.sort_values("PCT_CASTILLO"),
        x="PCT_CASTILLO", y="DEPARTAMENTO",
        orientation="h", color="GANADOR",
        color_discrete_map={
            "Pedro Castillo": "#c0392b",
            "Keiko Fujimori": "#2c3e50"
        },
        title="% de votos para Castillo por Departamento",
        hover_data=["PCT_FUJIMORI", "TOTAL_MESAS", "VOTOS_VALIDOS"]
    )
    fig_reg.update_layout(height=700)
    fig_reg.add_vline(x=50, line_dash="dash", line_color="gray",
                      annotation_text="50%")
    st.plotly_chart(fig_reg, use_container_width=True)

    st.subheader("📋 Tabla por departamento")
    st.dataframe(por_depa[[
        "DEPARTAMENTO","VOTOS_CASTILLO","VOTOS_FUJIMORI",
        "PCT_CASTILLO","PCT_FUJIMORI","GANADOR","TOTAL_MESAS"
    ]], use_container_width=True)

# ─────────────────────────────────
# TAB 3 — VISUALIZACIONES
# ─────────────────────────────────
with tab3:
    st.header("📈 Visualizaciones Detalladas")

    # Histograma % Castillo por mesa
    fig_hist = px.histogram(
        df_filtrado, x="PCT_CASTILLO", nbins=50,
        color_discrete_sequence=["#c0392b"],
        title=f"Distribución del % de votos para Castillo — {depa_sel}",
        labels={"PCT_CASTILLO": "% Votos Castillo por mesa"}
    )
    fig_hist.add_vline(x=50, line_dash="dash", line_color="black",
                       annotation_text="50%")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter nulos vs válidos
    fig_scat = px.scatter(
        por_depa, x="VOTOS_VALIDOS", y="VOTOS_NULOS",
        size="TOTAL_MESAS", color="GANADOR",
        hover_name="DEPARTAMENTO",
        title="Votos Válidos vs Nulos por Departamento",
        color_discrete_map={
            "Pedro Castillo": "#c0392b",
            "Keiko Fujimori": "#2c3e50"
        }
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    # Mesas ganadas por candidato
    mesas_gan = df["GANADOR_MESA"].value_counts().reset_index()
    mesas_gan.columns = ["Candidato", "Mesas"]
    fig_mesas = px.bar(
        mesas_gan, x="Candidato", y="Mesas", color="Candidato",
        color_discrete_map={
            "Pedro Castillo": "#c0392b",
            "Keiko Fujimori": "#2c3e50"
        },
        title="Número de mesas ganadas por candidato",
        text="Mesas"
    )
    fig_mesas.update_traces(textposition="outside")
    st.plotly_chart(fig_mesas, use_container_width=True)

# ─────────────────────────────────
# TAB 4 — MACHINE LEARNING
# ─────────────────────────────────
with tab4:
    st.header("🤖 Análisis de Machine Learning")

    st.markdown("""
    ### 🎯 Tipo de problema identificado
    - **Regresión:** Predecir votos de Castillo en base a variables de la mesa
    - **Clustering:** Agrupar mesas por comportamiento electoral similar
    """)

    # ── Regresión Lineal ──
    st.subheader("📐 Modelo 1: Regresión Lineal")

    df_ml = df[["VOTOS_TOTAL","VOTOS_VN","VOTOS_VB","VOTOS_P1"]].dropna()
    df_ml = df_ml[df_ml["VOTOS_TOTAL"] > 0]

    X = df_ml[["VOTOS_TOTAL","VOTOS_VN","VOTOS_VB"]]
    y = df_ml["VOTOS_P1"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    train_score = modelo.score(X_train, y_train)
    test_score  = modelo.score(X_test, y_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE", f"{rmse:.2f} votos")
    col3.metric("Mesas de prueba", f"{len(X_test):,}")

    fig_reg = px.scatter(
        x=y_test[:1000], y=y_pred[:1000],
        labels={"x": "Votos reales Castillo", "y": "Votos predichos"},
        title="Votos Reales vs Predichos (muestra 1000 mesas)",
        opacity=0.4, color_discrete_sequence=["#c0392b"]
    )
    fig_reg.add_shape(
        type="line", x0=0, y0=0,
        x1=int(y_test.max()), y1=int(y_test.max()),
        line=dict(color="blue", dash="dash")
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # Sobreajuste / Subajuste
    st.subheader("⚠️ Evaluación: Sobreajuste vs Subajuste")
    col4, col5 = st.columns(2)
    col4.metric("R² Entrenamiento", f"{train_score:.4f}")
    col5.metric("R² Prueba",        f"{test_score:.4f}")

    diferencia = abs(train_score - test_score)
    if diferencia < 0.02:
        st.success("✅ Modelo bien ajustado — diferencia mínima entre train y test")
    elif train_score > test_score + 0.1:
        st.warning("⚠️ Posible sobreajuste — funciona mejor en entrenamiento")
    else:
        st.warning("⚠️ Posible subajuste — no captura bien la relación")

    st.info(f"""
    **Interpretación:** El modelo explica el {r2*100:.1f}% de la variación 
    en votos de Castillo. Error promedio de {rmse:.1f} votos por mesa.
    
    **Limitación:** En análisis electoral real se necesitan variables 
    socioeconómicas, demográficas y geográficas para mayor precisión.
    """)

    st.markdown("---")

    # ── Clustering ──
    st.subheader("🔵 Modelo 2: Clustering K-Means")

    df_clus = df[["PCT_CASTILLO","PCT_FUJIMORI","VOTOS_VN"]].dropna()
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(df_clus)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clus = df_clus.copy()
    df_clus["CLUSTER"] = kmeans.fit_predict(X_sc)
    df_clus["CLUSTER"] = df_clus["CLUSTER"].map({
        0: "Grupo A — Dominio Castillo",
        1: "Grupo B — Dominio Fujimori",
        2: "Grupo C — Resultado mixto"
    })

    fig_clus = px.scatter(
        df_clus.sample(min(5000, len(df_clus))),
        x="PCT_CASTILLO", y="PCT_FUJIMORI",
        color="CLUSTER", opacity=0.5,
        title="Agrupamiento K-Means de mesas (K=3)",
        labels={
            "PCT_CASTILLO": "% Castillo",
            "PCT_FUJIMORI": "% Fujimori"
        },
        color_discrete_sequence=["#c0392b","#2c3e50","#f39c12"]
    )
    fig_clus.add_vline(x=50, line_dash="dash", line_color="gray")
    fig_clus.add_hline(y=50, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_clus, use_container_width=True)

    st.dataframe(
        df_clus["CLUSTER"].value_counts().reset_index(),
        use_container_width=True
    )

# ─────────────────────────────────
# TAB 5 — DATOS POR MESA
# ─────────────────────────────────
with tab5:
    st.header("📋 Consulta de Resultados por Mesa")
    st.write(f"Departamento: **{depa_sel}** | Mesas: **{len(df_filtrado):,}**")

    cols = [c for c in [
        "DEPARTAMENTO","PROVINCIA","DISTRITO","UBIGEO",
        "MESA_DE_VOTACION","VOTOS_P1","VOTOS_P2",
        "VOTOS_VB","VOTOS_VN","VOTOS_VALIDOS","GANADOR_MESA"
    ] if c in df_filtrado.columns]

    st.dataframe(df_filtrado[cols].head(500), use_container_width=True)

    csv_dl = df_filtrado[cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=csv_dl,
        file_name=f"onpe_{depa_sel.lower()}.csv",
        mime="text/csv"
    )

# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
    🗳️ Fuente oficial: ONPE — Plataforma Nacional de Datos Abiertos del Perú |
    Segunda Elección Presidencial 2021 | Uso académico
</div>
""", unsafe_allow_html=True)