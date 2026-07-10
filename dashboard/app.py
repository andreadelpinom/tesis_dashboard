import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador de Compañeros Deportivos", layout="wide")

# ──────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS (con cache para que no se recargue en cada clic)
# ──────────────────────────────────────────────────────────────

@st.cache_data
def cargar_datos():
    encuesta = pd.read_csv("data/encuesta_limpia.csv")
    clusters = pd.read_csv("data/estudiantes_clusters_ponderado.csv")
    similitud = pd.read_csv("data/similitud_ponderada.csv")
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


def construir_vector_nuevo(personalidad, horario, dias_sel, motivacion,
                            deportes_sel, actividad_sel, social_vals):
    vector = {c: 0 for grupo in GRUPOS.values() for c in grupo[0]}

    col_personalidad = next(c for c in GRUPOS["personalidad"][0] if c.endswith(personalidad))
    vector[col_personalidad] = 1

    col_horario = next(c for c in GRUPOS["horario"][0] if c.endswith(horario))
    vector[col_horario] = 1

    for d in dias_sel:
        col = next(c for c in GRUPOS["dias"][0] if c.endswith(d))
        vector[col] = 1

    col_motivacion = next(c for c in GRUPOS["motivacion"][0] if c.endswith(motivacion))
    vector[col_motivacion] = 1

    for d in deportes_sel:
        col = next(c for c in GRUPOS["deportes"][0] if c.endswith(d))
        vector[col] = 1

    for a in actividad_sel:
        col = next(c for c in GRUPOS["actividad"][0] if c.endswith(a))
        vector[col] = 1

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
    st.write(f"**Personalidad:** {nombre_personalidad(fila)}")
    st.write(f"**Horario preferido:** {nombre_horario(fila)}")
    st.write(f"**Días disponibles:** {dias_disponibles(fila)}")
    st.write(f"**Motivación principal:** {nombre_motivacion(fila)}")
    st.write(f"**Deportes que practica:** {deportes_practicados(fila)}")
    st.write(f"**Cluster:** {int(fila['cluster'])}")


def mostrar_tabla_top10(top10: pd.DataFrame):
    filas = []
    for _, r in top10.iterrows():
        fila_est = encuesta.loc[encuesta.id_estudiante == r["id_estudiante"]].iloc[0]
        filas.append({
            "ID": int(r["id_estudiante"]),
            "Similitud": round(r["similitud_ponderada"], 3),
            "Personalidad": nombre_personalidad(fila_est),
            "Horario": nombre_horario(fila_est),
            "Motivación": nombre_motivacion(fila_est),
            "Días": dias_disponibles(fila_est),
            "Cluster": int(fila_est["cluster"]),
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# 4. INTERFAZ
# ──────────────────────────────────────────────────────────────

st.title("🏅 Recomendador de Compañeros Deportivos")
st.caption(
    "Sistema basado en afinidad de preferencias para la integración social "
    "de estudiantes universitarios mediante actividades deportivas."
)

modo = st.sidebar.radio("¿Qué quieres hacer?", ["Buscar estudiante existente", "Soy un estudiante nuevo"])

if modo == "Buscar estudiante existente":
    st.header("Buscar estudiante existente")
    id_sel = st.selectbox("Selecciona el ID del estudiante", sorted(encuesta["id_estudiante"].tolist()))

    fila = encuesta.loc[encuesta.id_estudiante == id_sel].iloc[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"Perfil — Estudiante #{id_sel}")
        mostrar_perfil(fila)

    with col2:
        st.subheader("TOP-10 compañeros más compatibles")
        top10 = top10_para_id(id_sel)
        mostrar_tabla_top10(top10)

else:
    st.header("Ingresa tu perfil")
    st.write("Responde como en la encuesta original. Tus datos no se guardan.")

    col1, col2 = st.columns(2)

    with col1:
        personalidad = st.selectbox(
            "¿Cómo describirías tu personalidad?",
            ["Introvertida", "Extrovertida", "Ambidivertida"],
        )
        motivacion = st.selectbox(
            "¿Cuál es tu principal motivación para hacer deporte?",
            ["Competencia", "Diversión", "Mejorar condición física",
             "Mejorar salud", "Reducir estrés", "Socializar"],
        )
        horario = st.selectbox(
            "¿En qué horarios prefieres realizar actividades deportivas?",
            ["Mañana", "Mañana, Noche", "Mañana, Tarde", "Mañana, Tarde, Noche",
             "Noche", "Tarde", "Tarde, Noche"],
        )
        dias_sel = st.multiselect(
            "¿Qué días tienes disponibilidad?",
            ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"],
        )

    with col2:
        deportes_sel = st.multiselect(
            "¿Qué deportes practicas o te interesan?",
            ["Baile", "Basket", "Fútbol", "Gym/Fitness", "Natación",
             "Pádel", "Running", "Tenis", "Tenis de mesa", "Volley"],
        )
        actividad_sel = st.multiselect(
            "¿Qué tipo de actividades prefieres?",
            ["Al aire libre", "Bajo techo", "Competitivas",
             "Grupales", "Individuales", "Recreativas"],
        )
        st.write("**Perfil social** (escala 1 a 5):")
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

            st.success(f"Perteneces al **Cluster {cluster_asignado}** (asignado por afinidad con tus vecinos más cercanos).")
            st.subheader("Tu TOP-10 de compañeros más compatibles")
            mostrar_tabla_top10(top10)
