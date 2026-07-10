import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador de Compañeros Deportivos", layout="wide")

# Carpeta donde vive este mismo archivo app.py (sin importar desde dónde se ejecute)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ──────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS (con cache para que no se recargue en cada clic)
# ──────────────────────────────────────────────────────────────

@st.cache_data
def cargar_datos():
    encuesta = pd.read_csv(DATA_DIR / "encuesta_limpia.csv")
    clusters = pd.read_csv(DATA_DIR / "estudiantes_clusters_ponderado.csv")
    similitud = pd.read_csv(DATA_DIR / "similitud_ponderada.csv")
    encuesta["id_estudiante"] = encuesta.index
    encuesta = encuesta.merge(clusters, on="id_estudiante", how="left")
    return encuesta, similitud


encuesta, similitud = cargar_datos()

# ──────────────────────────────────────────────────────────────
# 2. DEFINICIÓN DE GRUPOS Y PESOS (idéntico a la Fase II de la tesis)
# ──────────────────────────────────────────────────────────────

GRUPOS = {
    "personalidad": (
        [c for c in encuesta.columns if c.startswith("¿Cómo describirías tu personal")],
        0.30,
    ),
    "horario": (
        [c for c in encuesta.columns if c.startswith("¿En qué horarios prefieres rea")],
        0.25,
    ),
    "dias": (
        [c for c in encuesta.columns if c.startswith("¿Qué días tienes disponib")],
        0.20,
    ),
    "motivacion": (
        [c for c in encuesta.columns if c.startswith("¿Cuál es tu principal motivaci")],
        0.15,
    ),
    "deportes": (
        [c for c in encuesta.columns if c.startswith("¿Qué deportes practicas o")],
        0.05,
    ),
    "actividad": (
        [c for c in encuesta.columns if c.startswith("¿Qué tipo de actividades")],
        0.03,
    ),
    "social": (
        [
            "Me considero una persona sociable.",
            "Me siento cómodo/a participando en actividades grupales.",
            "¿Qué tan fácil se te hace hacer nuevas amistades en la universidad?",
        ],
        0.02,
    ),
}

COLS_SOCIAL = GRUPOS["social"][0]
scaler = MinMaxScaler()
scaler.fit(encuesta[COLS_SOCIAL])

# Nombres y colores de los 7 clústeres (Tabla 2, Fase II de la tesis).
# Mostrar el nombre real (no solo el número) es clave para que alguien que no
# leyó la tesis entienda de inmediato a qué perfil de grupo pertenece.
CLUSTER_INFO = {
    0: ("Extrovertidos matutinos activos", "#7A1F35"),
    1: ("Ambidivertidos nocturnos orientados a la salud", "#3E6B8A"),
    2: ("Introvertidos matutinos orientados a la salud", "#2F7D5B"),
    3: ("Ambidivertidos matutinos activos", "#B5762A"),
    4: ("Ambidivertidos vespertinos activos", "#6B4E9E"),
    5: ("Introvertidos vespertinos orientados a la salud", "#46807C"),
    6: ("Ambidivertidos vespertino-nocturnos orientados al bienestar", "#8A4A2F"),
}


def nombre_personalidad(row):
    for c in GRUPOS["personalidad"][0]:
        if row[c] == 1:
            return c.split("_")[-1]
    return "N/D"


def nombre_horario(row):
    for c in GRUPOS["horario"][0]:
        if row[c] == 1:
            return c.split("_")[-1]
    return "N/D"


def nombre_motivacion(row):
    for c in GRUPOS["motivacion"][0]:
        if row[c] == 1:
            return c.split("_")[-1]
    return "N/D"


def dias_disponibles(row):
    dias = [c.split("_")[-1] for c in GRUPOS["dias"][0] if row[c] == 1]
    return ", ".join(dias) if dias else "N/D"


def deportes_practicados(row):
    deportes = [c.split("_")[-1] for c in GRUPOS["deportes"][0] if row[c] == 1]
    return ", ".join(deportes) if deportes else "Ninguno registrado"


def badge_cluster(cluster_id: int) -> str:
    nombre, color = CLUSTER_INFO.get(int(cluster_id), (f"Cluster {cluster_id}", "#5B5458"))
    return (
        f'<span style="background:{color}1A; color:{color}; border:1px solid {color}55; '
        f'padding:0.25rem 0.65rem; border-radius:999px; font-size:0.82rem; font-weight:600; '
        f'display:inline-flex; align-items:center; gap:0.4rem;">'
        f'<span style="width:8px; height:8px; border-radius:50%; background:{color}; display:inline-block;"></span>'
        f'{int(cluster_id)} · {nombre}</span>'
    )


# ──────────────────────────────────────────────────────────────
# 3. SIMILITUD PONDERADA (misma fórmula de la Fase II, verificada)
# ──────────────────────────────────────────────────────────────

def similitud_ponderada_vector(vector_a: dict, fila_b: pd.Series) -> float:
    """Calcula la similitud coseno ponderada entre un vector nuevo (dict de 0/1
    o valores Likert normalizados) y una fila existente de la encuesta."""
    total = 0.0
    for _, (cols, peso) in GRUPOS.items():
        va = np.array([vector_a[c] for c in cols], dtype=float).reshape(1, -1)
        vb = fila_b[cols].values.astype(float).reshape(1, -1)
        if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
            sim = 0.0
        else:
            sim = cosine_similarity(va, vb)[0, 0]
        total += peso * sim
    return total


def _col_por_categoria_exacta(cols, categoria):
    """Encuentra la columna cuya categoría (después del primer '_') coincide
    EXACTAMENTE con la seleccionada. Usar .endswith() aquí es incorrecto porque
    categorías compuestas como 'Mañana, Noche' también terminan en 'Noche',
    lo que causaría falsos positivos con la categoría simple 'Noche'."""
    for c in cols:
        sufijo = c.split("_", 1)[-1]
        if sufijo == categoria:
            return c
    raise ValueError(f"No se encontró la categoría '{categoria}' entre las columnas: {cols}")


def construir_vector_nuevo(personalidad, horario, dias_sel, motivacion,
                            deportes_sel, actividad_sel, social_vals):
    vector = {c: 0 for grupo in GRUPOS.values() for c in grupo[0]}

    vector[_col_por_categoria_exacta(GRUPOS["personalidad"][0], personalidad)] = 1
    vector[_col_por_categoria_exacta(GRUPOS["horario"][0], horario)] = 1

    for d in dias_sel:
        vector[_col_por_categoria_exacta(GRUPOS["dias"][0], d)] = 1

    vector[_col_por_categoria_exacta(GRUPOS["motivacion"][0], motivacion)] = 1

    for d in deportes_sel:
        vector[_col_por_categoria_exacta(GRUPOS["deportes"][0], d)] = 1

    for a in actividad_sel:
        vector[_col_por_categoria_exacta(GRUPOS["actividad"][0], a)] = 1

    # Normalizar los 3 Likert sociales con el mismo scaler ajustado sobre los 158 estudiantes
    social_norm = scaler.transform([social_vals])[0]
    for col, val in zip(COLS_SOCIAL, social_norm):
        vector[col] = val

    return vector


def top10_para_id(id_estudiante: int) -> pd.DataFrame:
    sub = similitud[(similitud.id_i == id_estudiante) | (similitud.id_j == id_estudiante)].copy()
    sub["vecino"] = np.where(sub.id_i == id_estudiante, sub.id_j, sub.id_i)
    sub = sub.sort_values("similitud_ponderada", ascending=False).head(10)
    return sub[["vecino", "similitud_ponderada"]].rename(columns={"vecino": "id_estudiante"})


def top10_para_vector(vector_nuevo: dict) -> pd.DataFrame:
    sims = encuesta.apply(lambda fila: similitud_ponderada_vector(vector_nuevo, fila), axis=1)
    resultado = pd.DataFrame({
        "id_estudiante": encuesta["id_estudiante"],
        "similitud_ponderada": sims,
    })
    return resultado.sort_values("similitud_ponderada", ascending=False).head(10)


def asignar_cluster_nuevo(top_vecinos: pd.DataFrame, k_vecinos: int = 15) -> int:
    """Asigna cluster por voto mayoritario entre los k vecinos más afines
    (necesario porque el clustering se hizo sobre la matriz de afinidad,
    no sobre las variables originales, así que no existe un centroide
    al cual comparar directamente a un estudiante nuevo)."""
    vecinos_id = top_vecinos["id_estudiante"].head(k_vecinos)
    clusters_vecinos = encuesta.set_index("id_estudiante").loc[vecinos_id, "cluster"]
    return int(clusters_vecinos.mode()[0])


def mostrar_perfil(fila: pd.Series):
    campos = [
        ("Personalidad", nombre_personalidad(fila)),
        ("Horario preferido", nombre_horario(fila)),
        ("Días disponibles", dias_disponibles(fila)),
        ("Motivación principal", nombre_motivacion(fila)),
        ("Deportes que practica", deportes_practicados(fila)),
    ]
    filas_html = "".join(
        f'<div style="display:flex; justify-content:space-between; gap:1rem; '
        f'padding:0.45rem 0; border-bottom:1px solid #EFE7E8;">'
        f'<span style="color:#5B5458; font-size:0.82rem; text-transform:uppercase; '
        f'letter-spacing:0.02em;">{label}</span>'
        f'<span style="color:#211A1C; font-weight:600; text-align:right;">{valor}</span>'
        f'</div>'
        for label, valor in campos
    )
    st.markdown(filas_html, unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(badge_cluster(fila["cluster"]), unsafe_allow_html=True)


def _fila_tabla_html(fila: dict, similitud_max: float = 1.0) -> str:
    porcentaje = max(0.0, min(100.0, (fila["Similitud"] / similitud_max) * 100))
    _, color_cluster = CLUSTER_INFO.get(fila["Cluster"], ("", "#5B5458"))
    return (
        '<tr>'
        f'<td style="padding:0.6rem 0.8rem; font-weight:600; color:#211A1C;">{fila["ID"]}</td>'
        '<td style="padding:0.6rem 0.8rem; min-width:160px;">'
        '<div style="display:flex; align-items:center; gap:0.6rem;">'
        '<div style="flex:1; background:#EFE2E5; border-radius:999px; height:8px; overflow:hidden;">'
        f'<div style="width:{porcentaje:.1f}%; background:#7A1F35; height:100%; border-radius:999px;"></div>'
        '</div>'
        f'<span style="font-weight:700; color:#7A1F35; font-size:0.85rem; white-space:nowrap;">{porcentaje:.0f}%</span>'
        '</div></td>'
        f'<td style="padding:0.6rem 0.8rem; color:#211A1C;">{fila["Personalidad"]}</td>'
        f'<td style="padding:0.6rem 0.8rem; color:#211A1C;">{fila["Horario"]}</td>'
        f'<td style="padding:0.6rem 0.8rem; color:#211A1C;">{fila["Motivación"]}</td>'
        f'<td style="padding:0.6rem 0.8rem; color:#211A1C;">{fila["Días"]}</td>'
        '<td style="padding:0.6rem 0.8rem;">'
        f'<span style="display:inline-flex; align-items:center; gap:0.35rem; font-size:0.85rem; color:#211A1C;">'
        f'<span style="width:9px; height:9px; border-radius:50%; background:{color_cluster}; display:inline-block;"></span>'
        f'{fila["Cluster"]}</span></td>'
        '</tr>'
    )


def _kpi_card(label: str, valor: str) -> str:
    return (
        f'<div style="background:#FFFFFF; border:1px solid #E8DEE0; border-radius:10px; '
        f'padding:0.9rem 1.1rem; flex:1;">'
        f'<div style="color:#5B5458; font-size:0.78rem; font-weight:600; margin-bottom:0.35rem; '
        f'line-height:1.3;">{label}</div>'
        f'<div style="color:#211A1C; font-size:1.4rem; font-weight:700; line-height:1.25; '
        f'word-break:break-word;">{valor}</div>'
        f'</div>'
    )


def mostrar_tabla_top10(top10: pd.DataFrame):
    filas = []
    for _, r in top10.iterrows():
        fila_est = encuesta.loc[encuesta.id_estudiante == r["id_estudiante"]].iloc[0]
        filas.append({
            "ID": int(r["id_estudiante"]),
            "Similitud": round(float(r["similitud_ponderada"]), 3),
            "Personalidad": nombre_personalidad(fila_est),
            "Horario": nombre_horario(fila_est),
            "Motivación": nombre_motivacion(fila_est),
            "Días": dias_disponibles(fila_est),
            "Cluster": int(fila_est["cluster"]),
        })
    df_tabla = pd.DataFrame(filas)

    # KPIs con lenguaje directo: se evita jerga técnica ("similitud ponderada")
    # y se usa porcentaje en vez de decimales (0.729 es menos intuitivo que 73%).
    # Se usan tarjetas HTML propias (no st.metric) porque st.metric trunca con "…"
    # cuando la etiqueta o el valor son largos, cortando información importante.
    mejor = df_tabla.iloc[0]
    cluster_top_id = int(df_tabla["Cluster"].mode()[0])
    nombre_cluster_top, _ = CLUSTER_INFO.get(cluster_top_id, (f"Cluster {cluster_top_id}", "#5B5458"))

    kpis_html = (
        '<div style="display:flex; gap:0.8rem; flex-wrap:wrap; margin-bottom:0.8rem;">'
        + _kpi_card("Compatibilidad promedio con tus 10 recomendados", f"{df_tabla['Similitud'].mean() * 100:.0f}%")
        + _kpi_card("Tu compañero más compatible", f"#{int(mejor['ID'])} · {mejor['Similitud'] * 100:.0f}%")
        + _kpi_card("Grupo con más compañeros afines", nombre_cluster_top)
        + '</div>'
    )
    st.markdown(kpis_html, unsafe_allow_html=True)

    filas_html = "".join(_fila_tabla_html(f) for f in filas)
    # IMPORTANTE: el HTML va en una sola línea, sin saltos de línea ni indentación.
    # Streamlit interpreta líneas en blanco dentro de un bloque markdown como
    # separadores de párrafo, incluso con unsafe_allow_html=True; eso rompe una
    # tabla en varios fragmentos y deja etiquetas como "</tbody>" visibles como texto.
    tabla_html = (
        '<div style="overflow-x:auto; border:1px solid #E8DEE0; border-radius:10px;">'
        '<table style="width:100%; border-collapse:collapse; font-family:\'Inter\',sans-serif; font-size:0.88rem;">'
        '<thead><tr style="background:#FAF7F7; border-bottom:1px solid #E8DEE0;">'
        '<th style="padding:0.6rem 0.8rem; text-align:left; color:#5B5458; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">ID</th>'
        '<th style="padding:0.6rem 0.8rem; text-align:left; color:#5B5458; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">Compatibilidad</th>'
        '<th style="padding:0.6rem 0.8rem; text-align:left; color:#5B5458; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">Personalidad</th>'
        '<th style="padding:0.6rem 0.8rem; text-align:left; color:#5B5458; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">Horario</th>'
        '<th style="padding:0.6rem 0.8rem; text-align:left; color:#5B5458; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">Motivación</th>'
        '<th style="padding:0.6rem 0.8rem; text-align:left; color:#5B5458; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">Días</th>'
        '<th style="padding:0.6rem 0.8rem; text-align:left; color:#5B5458; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;">Cluster</th>'
        '</tr></thead><tbody>' + filas_html + '</tbody></table></div>'
    )
    st.markdown(tabla_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 4. ESTILOS E IDENTIDAD VISUAL (colores institucionales UEES)
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #FAF7F7;
    }

    /* Barra superior institucional */
    .uees-header {
        background: linear-gradient(90deg, #7A1F35 0%, #5C1626 100%);
        padding: 1.4rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .uees-header .mark {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.8rem;
        color: #FFFFFF;
        background: rgba(255,255,255,0.14);
        border-radius: 8px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .uees-header .titles h1 {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .uees-header .titles p {
        color: rgba(255,255,255,0.85);
        font-size: 0.85rem;
        margin: 0.15rem 0 0 0;
    }

    /* Encabezados de sección dentro de tarjetas */
    .card-eyebrow {
        color: #7A1F35;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }

    /* Botones primarios */
    .stButton > button[kind="primary"] {
        background-color: #7A1F35;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #5C1626;
    }

    /* Sidebar: segmented control para el radio de modo */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        background: #F3E4E8;
        padding: 0.3rem;
        border-radius: 10px;
        gap: 0.2rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent;
        border-radius: 8px;
        padding: 0.4rem 0.6rem;
    }
    /* Punto del radio: Streamlit lo pinta rojo por defecto; lo forzamos a granate */
    section[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child {
        border-color: #7A1F35 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child > div {
        background-color: #7A1F35 !important;
    }
    /* Texto de la opción seleccionada en negrita, la no seleccionada más discreta */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
        color: #211A1C;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] p {
        color: #5B5458;
        font-size: 0.92rem;
    }

    /* Eyebrow del sidebar (encabezado "MENÚ") */
    .sidebar-eyebrow {
        color: #7A1F35;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 0.5rem 0;
    }

    /* Expander del sidebar: título en granate, borde consistente con las tarjetas */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid #E8DEE0;
        border-radius: 10px;
        background: #FFFFFF;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #5C1626;
        font-weight: 600;
    }

    h2, h3 {
        color: #5C1626;
    }

    /* Métricas: números tabulares para alineación consistente */
    [data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
        color: #211A1C;
    }

    /* Sliders: el punto (thumb) usa un atributo ARIA estable (role="slider"),
    por eso es seguro sobreescribirlo. El riel de color, en cambio, no tiene
    un selector estable entre versiones de Streamlit — forzarlo con selectores
    de posición (nth-child) rompía otros elementos internos del slider, así que
    se dejó fuera. Ese color depende del tema configurado en config.toml. */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #7A1F35 !important;
        border-color: #7A1F35 !important;
    }

    /* Multiselect: etiquetas seleccionadas en color institucional */
    span[data-baseweb="tag"] {
        background-color: #7A1F35 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="uees-header">
    <div class="mark">U</div>
    <div class="titles">
        <h1>Recomendador de Compañeros Deportivos</h1>
        <p>Universidad de Especialidades Espíritu Santo — Sistema de recomendación por afinidad</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 5. SIDEBAR
# ──────────────────────────────────────────────────────────────

st.sidebar.markdown('<div class="sidebar-eyebrow">MENÚ</div>', unsafe_allow_html=True)
modo = st.sidebar.radio(
    "¿Qué quieres hacer?",
    ["Buscar estudiante existente", "Soy un estudiante nuevo"],
    label_visibility="collapsed",
)

with st.sidebar.expander("¿Cómo funciona?"):
    st.write(
        "El sistema calcula la afinidad entre estudiantes usando similitud "
        "coseno ponderada sobre 7 grupos de variables: personalidad (30%), "
        "horario (25%), días disponibles (20%), motivación (15%), deportes "
        "practicados (5%), tipo de actividad (3%) y perfil social (2%)."
    )
    st.write(
        "Si eres un estudiante nuevo, tu perfil se compara en tiempo real "
        "contra los 158 estudiantes existentes, y tu cluster se asigna por "
        "afinidad con tus 15 vecinos más parecidos."
    )

# ──────────────────────────────────────────────────────────────
# 6. INTERFAZ PRINCIPAL
# ──────────────────────────────────────────────────────────────

if modo == "Buscar estudiante existente":
    st.header("Buscar estudiante existente")
    id_sel = st.selectbox("Selecciona el ID del estudiante", sorted(encuesta["id_estudiante"].tolist()))

    fila = encuesta.loc[encuesta.id_estudiante == id_sel].iloc[0]

    with st.container(border=True):
        st.markdown('<div class="card-eyebrow">PERFIL</div>', unsafe_allow_html=True)
        st.subheader(f"Estudiante #{id_sel}")
        mostrar_perfil(fila)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-eyebrow">RECOMENDACIONES</div>', unsafe_allow_html=True)
        st.subheader("TOP-10 compañeros más compatibles")
        top10 = top10_para_id(id_sel)
        mostrar_tabla_top10(top10)

else:
    st.header("Ingresa tu perfil")
    st.caption("Responde como en la encuesta original. Tus datos no se guardan.")

    with st.container(border=True):
        st.markdown('<div class="card-eyebrow">PERFIL PERSONAL</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            personalidad = st.selectbox(
                "¿Cómo describirías tu personalidad?",
                ["Introvertida", "Extrovertida", "Ambidivertida"],
            )
        with col2:
            motivacion = st.selectbox(
                "¿Cuál es tu principal motivación para hacer deporte?",
                ["Competencia", "Diversión", "Mejorar condición física",
                 "Mejorar salud", "Reducir estrés", "Socializar"],
            )

    with st.container(border=True):
        st.markdown('<div class="card-eyebrow">DISPONIBILIDAD</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            horario = st.selectbox(
                "¿En qué horarios prefieres realizar actividades deportivas?",
                ["Mañana", "Mañana, Noche", "Mañana, Tarde", "Mañana, Tarde, Noche",
                 "Noche", "Tarde", "Tarde, Noche"],
            )
        with col2:
            dias_sel = st.multiselect(
                "¿Qué días tienes disponibilidad?",
                ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"],
            )

    with st.container(border=True):
        st.markdown('<div class="card-eyebrow">PREFERENCIAS DEPORTIVAS</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            deportes_sel = st.multiselect(
                "¿Qué deportes practicas o te interesan?",
                ["Baile", "Basket", "Fútbol", "Gym/Fitness", "Natación",
                 "Pádel", "Running", "Tenis", "Tenis de mesa", "Volley"],
            )
        with col2:
            actividad_sel = st.multiselect(
                "¿Qué tipo de actividades prefieres?",
                ["Al aire libre", "Bajo techo", "Competitivas",
                 "Grupales", "Individuales", "Recreativas"],
            )

    with st.container(border=True):
        st.markdown('<div class="card-eyebrow">PERFIL SOCIAL</div>', unsafe_allow_html=True)
        st.caption("Escala de 1 (nada de acuerdo) a 5 (totalmente de acuerdo)")
        s1 = st.slider("Me considero una persona sociable.", 1, 5, 3)
        s2 = st.slider("Me siento cómodo/a participando en actividades grupales.", 1, 5, 3)
        s3 = st.slider("¿Qué tan fácil se te hace hacer nuevas amistades?", 1, 5, 3)

    if st.button("Obtener mis recomendaciones", type="primary"):
        if not dias_sel:
            st.warning("Selecciona al menos un día de disponibilidad.")
        else:
            vector_nuevo = construir_vector_nuevo(
                personalidad, horario, dias_sel, motivacion,
                deportes_sel, actividad_sel, [s1, s2, s3],
            )
            with st.spinner("Calculando afinidad con los 158 estudiantes..."):
                top10 = top10_para_vector(vector_nuevo)
                cluster_asignado = asignar_cluster_nuevo(top10)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown(badge_cluster(cluster_asignado), unsafe_allow_html=True)
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="card-eyebrow">RECOMENDACIONES</div>', unsafe_allow_html=True)
                st.subheader("Tu TOP-10 de compañeros más compatibles")
                mostrar_tabla_top10(top10)
