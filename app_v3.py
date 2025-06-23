# Guardar como app_v3.py (sugiero un nuevo nombre para mantener versiones anteriores)
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Aplicación de Balance Metalúrgico de Cuatro Productos")

# --- INPUTS ---
st.header("1. Datos de Entrada")
peso_cabeza_input = st.number_input("Peso Total de Cabeza (ton)", value=4000.0, format="%.2f")

st.subheader("Leyes de Minerales (%)")
col1, col2, col3, col4, col5 = st.columns(5) # 5 columnas para 5 productos

# Valores de ejemplo para las leyes
with col1:
    st.write("**Cabeza**")
    ley_cabeza_pb = st.number_input("Pb en Cabeza", value=5.5, format="%.1f", key="cabeza_pb")
    ley_cabeza_zn = st.number_input("Zn en Cabeza", value=7.5, format="%.1f", key="cabeza_zn")
    ley_cabeza_cu = st.number_input("Cu en Cabeza", value=0.8, format="%.2f", key="cabeza_cu")
    # Ley de estaño eliminada

with col2:
    st.write("**Conc.Pb**")
    ley_conc_pb_pb = st.number_input("Pb en Conc.Pb", value=68.0, format="%.1f", key="conc_pb_pb")
    ley_conc_pb_zn = st.number_input("Zn en Conc.Pb", value=5.0, format="%.1f", key="conc_pb_zn")
    ley_conc_pb_cu = st.number_input("Cu en Conc.Pb", value=0.5, format="%.2f", key="conc_pb_cu")
    # Ley de estaño eliminada

with col3:
    st.write("**Conc.Zn**")
    ley_conc_zn_pb = st.number_input("Pb en Conc.Zn", value=1.0, format="%.1f", key="conc_zn_pb")
    ley_conc_zn_zn = st.number_input("Zn en Conc.Zn", value=55.0, format="%.1f", key="conc_zn_zn")
    ley_conc_zn_cu = st.number_input("Cu en Conc.Zn", value=0.2, format="%.2f", key="conc_zn_cu")
    # Ley de estaño eliminada

with col4:
    st.write("**Conc.Cu**") # Nuevo Producto
    ley_conc_cu_pb = st.number_input("Pb en Conc.Cu", value=1.5, format="%.1f", key="conc_cu_pb")
    ley_conc_cu_zn = st.number_input("Zn en Conc.Cu", value=2.0, format="%.1f", key="conc_cu_zn")
    ley_conc_cu_cu = st.number_input("Cu en Conc.Cu", value=25.0, format="%.1f", key="conc_cu_cu")
    # Ley de estaño eliminada

with col5:
    st.write("**Relave**")
    ley_relave_pb = st.number_input("Pb en Relave", value=0.2, format="%.1f", key="relave_pb")
    ley_relave_zn = st.number_input("Zn en Relave", value=0.6, format="%.1f", key="relave_zn")
    ley_relave_cu = st.number_input("Cu en Relave", value=0.05, format="%.2f", key="relave_cu")
    # Ley de estaño eliminada

# Recopilar las leyes en el diccionario (Sn eliminado)
leyes = {
    "Cabeza": {"Pb": ley_cabeza_pb, "Zn": ley_cabeza_zn, "Cu": ley_cabeza_cu},
    "Conc.Pb": {"Pb": ley_conc_pb_pb, "Zn": ley_conc_pb_zn, "Cu": ley_conc_pb_cu},
    "Conc.Zn": {"Pb": ley_conc_zn_pb, "Zn": ley_conc_zn_zn, "Cu": ley_conc_zn_cu},
    "Conc.Cu": {"Pb": ley_conc_cu_pb, "Zn": ley_conc_cu_zn, "Cu": ley_conc_cu_cu},
    "Relave": {"Pb": ley_relave_pb, "Zn": ley_relave_zn, "Cu": ley_relave_cu}
}

# --- LÓGICA DE BALANCE (RESOLUCIÓN DEL SISTEMA DE ECUACIONES) ---
# Convertir leyes a decimales para los cálculos
ley_cabeza_pb_dec = leyes["Cabeza"]["Pb"] / 100
ley_cabeza_zn_dec = leyes["Cabeza"]["Zn"] / 100
ley_cabeza_cu_dec = leyes["Cabeza"]["Cu"] / 100
ley_conc_pb_pb_dec = leyes["Conc.Pb"]["Pb"] / 100
ley_conc_pb_zn_dec = leyes["Conc.Pb"]["Zn"] / 100
ley_conc_pb_cu_dec = leyes["Conc.Pb"]["Cu"] / 100
ley_conc_zn_pb_dec = leyes["Conc.Zn"]["Pb"] / 100
ley_conc_zn_zn_dec = leyes["Conc.Zn"]["Zn"] / 100
ley_conc_zn_cu_dec = leyes["Conc.Zn"]["Cu"] / 100
ley_conc_cu_pb_dec = leyes["Conc.Cu"]["Pb"] / 100
ley_conc_cu_zn_dec = leyes["Conc.Cu"]["Zn"] / 100
ley_conc_cu_cu_dec = leyes["Conc.Cu"]["Cu"] / 100
ley_relave_pb_dec = leyes["Relave"]["Pb"] / 100
ley_relave_zn_dec = leyes["Relave"]["Zn"] / 100
ley_relave_cu_dec = leyes["Relave"]["Cu"] / 100

# Definir la matriz A y el vector B para el sistema Ax = B
# Incógnitas: x[0] = Peso Conc.Pb, x[1] = Peso Conc.Zn, x[2] = Peso Conc.Cu, x[3] = Peso Relave

# Ecuación 1 (Balance de Masa General): P_concPb + P_concZn + P_concCu + P_relave = P_cabeza
row1 = [1, 1, 1, 1]
b1 = peso_cabeza_input

# Ecuación 2 (Balance de Pb): P_concPb*LeyPb_concPb + ... + P_relave*LeyPb_relave = P_cabeza*LeyPb_cabeza
row2 = [ley_conc_pb_pb_dec, ley_conc_zn_pb_dec, ley_conc_cu_pb_dec, ley_relave_pb_dec]
b2 = peso_cabeza_input * ley_cabeza_pb_dec

# Ecuación 3 (Balance de Zn): P_concPb*LeyZn_concPb + ... + P_relave*LeyZn_relave = P_cabeza*LeyZn_cabeza
row3 = [ley_conc_pb_zn_dec, ley_conc_zn_zn_dec, ley_conc_cu_zn_dec, ley_relave_zn_dec]
b3 = peso_cabeza_input * ley_cabeza_zn_dec

# Ecuación 4 (Balance de Cu): P_concPb*LeyCu_concPb + ... + P_relave*LeyCu_relave = P_cabeza*LeyCu_cabeza
row4 = [ley_conc_pb_cu_dec, ley_conc_zn_cu_dec, ley_conc_cu_cu_dec, ley_relave_cu_dec]
b4 = peso_cabeza_input * ley_cabeza_cu_dec

# Construir la matriz A (4x4)
A = np.array([
    row1,
    row2,
    row3,
    row4
])

# Construir el vector B (4 elementos)
B = np.array([b1, b2, b3, b4])

peso_conc_pb, peso_conc_zn, peso_conc_cu, peso_relave = 0, 0, 0, 0 # Inicializar para evitar errores
try:
    x = np.linalg.solve(A, B)

    peso_conc_pb = x[0]
    peso_conc_zn = x[1]
    peso_conc_cu = x[2]
    peso_relave = x[3]

    if any(p < -1e-6 for p in x):
        st.warning("Uno o más pesos calculados son negativos, lo que puede indicar una inconsistencia en los datos de entrada (leyes).")
except np.linalg.LinAlgError:
    st.error("No se pudo resolver el sistema de ecuaciones. Esto puede deberse a que las leyes proporcionadas son inconsistentes o el sistema es singular.")
    st.info("Asegúrese de que hay suficiente diferencia entre las leyes para permitir un balance y que los datos no son linealmente dependientes.")
    st.stop() # Detiene la ejecución si hay error

# --- CONSTRUCCIÓN Y CÁLCULO COMPLETO DE LA TABLA ---

products_data = {
    "Productos": ["Cabeza", "Conc.Pb", "Conc.Zn", "Conc.Cu", "Relave"],
    "Peso": [peso_cabeza_input, peso_conc_pb, peso_conc_zn, peso_conc_cu, peso_relave],
    "Leyes %Pb": [leyes["Cabeza"]["Pb"], leyes["Conc.Pb"]["Pb"], leyes["Conc.Zn"]["Pb"], leyes["Conc.Cu"]["Pb"], leyes["Relave"]["Pb"]],
    "Leyes %Zn": [leyes["Cabeza"]["Zn"], leyes["Conc.Pb"]["Zn"], leyes["Conc.Zn"]["Zn"], leyes["Conc.Cu"]["Zn"], leyes["Relave"]["Zn"]],
    "Leyes %Cu": [leyes["Cabeza"]["Cu"], leyes["Conc.Pb"]["Cu"], leyes["Conc.Zn"]["Cu"], leyes["Conc.Cu"]["Cu"], leyes["Relave"]["Cu"]],
    # Leyes %Sn eliminada
}

df = pd.DataFrame(products_data)

# %Peso
df['%Peso'] = (df['Peso'] / peso_cabeza_input) * 100

# Contenido metálico (Peso * Ley / 100)
df['Contenido metálico Pb'] = df['Peso'] * (df['Leyes %Pb'] / 100)
df['Contenido metálico Zn'] = df['Peso'] * (df['Leyes %Zn'] / 100)
df['Contenido metálico Cu'] = df['Peso'] * (df['Leyes %Cu'] / 100)
# Contenido metálico Sn eliminado

# Obtener el total de contenido metálico en la Cabeza para la distribución
total_pb_en_cabeza = df.loc[df['Productos'] == 'Cabeza', 'Contenido metálico Pb'].iloc[0]
total_zn_en_cabeza = df.loc[df['Productos'] == 'Cabeza', 'Contenido metálico Zn'].iloc[0]
total_cu_en_cabeza = df.loc[df['Productos'] == 'Cabeza', 'Contenido metálico Cu'].iloc[0]

# % Distribución
df['%Distrib. Pb'] = (df['Contenido metálico Pb'] / total_pb_en_cabeza) * 100 if total_pb_en_cabeza != 0 else 0
df['%Distrib. Zn'] = (df['Contenido metálico Zn'] / total_zn_en_cabeza) * 100 if total_zn_en_cabeza != 0 else 0
df['%Distrib. Cu'] = (df['Contenido metálico Cu'] / total_cu_en_cabeza) * 100 if total_cu_en_cabeza != 0 else 0
# % Distribución Sn eliminado

# Cálculo del Ratio (Factor de Concentración de Masa)
df['Ratio'] = np.nan
if peso_conc_pb != 0:
    df.loc[df['Productos'] == 'Conc.Pb', 'Ratio'] = peso_cabeza_input / peso_conc_pb
if peso_conc_zn != 0:
    df.loc[df['Productos'] == 'Conc.Zn', 'Ratio'] = peso_cabeza_input / peso_conc_zn
if peso_conc_cu != 0:
    df.loc[df['Productos'] == 'Conc.Cu', 'Ratio'] = peso_cabeza_input / peso_conc_cu


# --- FORMATEO Y PRESENTACIÓN FINAL DE LA TABLA ---
df_final = df[[
    "Productos", "Peso", "%Peso",
    "Leyes %Pb", "Leyes %Zn", "Leyes %Cu", # Leyes %Sn eliminada
    "Contenido metálico Pb", "Contenido metálico Zn", "Contenido metálico Cu", # Contenido metálico Sn eliminado
    "%Distrib. Pb", "%Distrib. Zn", "%Distrib. Cu", # %Distrib. Sn eliminado
    "Ratio"
]]

df_final.columns = [
    "Productos", "Peso", "%Peso",
    "Leyes\n%Pb", "Leyes\n%Zn", "Leyes\n%Cu", # Leyes\n%Sn eliminada
    "Contenido metálico\nPb", "Contenido metálico\nZn", "Contenido metálico\nCu", # Contenido metálico\nSn eliminado
    "%Distrib.\nPb", "%Distrib.\nZn", "%Distrib.\nCu", # %Distrib.\nSn eliminado
    "Ratio"
]

# Formato para visualización en Streamlit
df_display = df_final.copy()
for col in df_display.columns:
    if 'Peso' in col or 'Contenido' in col:
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.3f}" if pd.notna(x) else "")
    elif '%' in col and "Leyes" not in col: # Aplicar a %Peso y %Distrib.
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}%" if pd.notna(x) else "")
    elif 'Leyes' in col: # Leyes Pb, Zn, Cu
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    elif 'Ratio' in col:
        df_display[col] = df_display[col].apply(lambda x: f"{x:,.3f}" if pd.notna(x) else "")
    else: # Para productos
        df_display[col] = df_display[col].apply(lambda x: str(x).replace('\n', ' '))


# --- SALIDAS EN STREAMLIT ---
st.header("2. Tabla de Balance Metalúrgico")
st.dataframe(df_display, hide_index=True, use_container_width=True) # Streamlit renderiza DataFrames bellamente

st.header("3. Verificaciones de Balance")
st.write(f"**Suma de Pesos de Productos (Conc.Pb + Conc.Zn + Conc.Cu + Relave):** {df_final.loc[1:4, 'Peso'].sum():,.2f}")
st.write(f"**Peso de Cabeza (input):** {df_final.loc[0, 'Peso']:,.2f}")
st.write(f"**Diferencia de Pesos:** {df_final.loc[0, 'Peso'] - df_final.loc[1:4, 'Peso'].sum():,.2f}")


st.write(f"**Balance de Pb (Total Pb en productos):** {df_final.loc[1:4, 'Contenido metálico\nPb'].sum():,.3f}")
st.write(f"**Balance de Pb (Total Pb en Cabeza):** {df_final.loc[0, 'Contenido metálico\nPb']:,.3f}")
st.write(f"**Diferencia Pb:** {df_final.loc[0, 'Contenido metálico\nPb'] - df_final.loc[1:4, 'Contenido metálico\nPb'].sum():,.3f}")

st.write(f"**Balance de Zn (Total Zn en productos):** {df_final.loc[1:4, 'Contenido metálico\nZn'].sum():,.3f}")
st.write(f"**Balance de Zn (Total Zn en Cabeza):** {df_final.loc[0, 'Contenido metálico\nZn']:,.3f}")
st.write(f"**Diferencia Zn:** {df_final.loc[0, 'Contenido metálico\nZn'] - df_final.loc[1:4, 'Contenido metálico\nZn'].sum():,.3f}")

st.write(f"**Balance de Cu (Total Cu en productos):** {df_final.loc[1:4, 'Contenido metálico\nCu'].sum():,.3f}")
st.write(f"**Balance de Cu (Total Cu en Cabeza):** {df_final.loc[0, 'Contenido metálico\nCu']:,.3f}")
st.write(f"**Diferencia Cu:** {df_final.loc[0, 'Contenido metálico\nCu'] - df_final.loc[1:4, 'Contenido metálico\nCu'].sum():,.3f}")

# La nota sobre el Sn y sus verificaciones han sido eliminadas

st.write(f"\n**Suma %Distrib. Pb (Conc.Pb + Conc.Zn + Conc.Cu + Relave):** {df_final.loc[1:4, '%Distrib.\nPb'].sum():,.2f} %")
st.write(f"**Suma %Distrib. Zn (Conc.Pb + Conc.Zn + Conc.Cu + Relave):** {df_final.loc[1:4, '%Distrib.\nZn'].sum():,.2f} %")
st.write(f"**Suma %Distrib. Cu (Conc.Pb + Conc.Zn + Conc.Cu + Relave):** {df_final.loc[1:4, '%Distrib.\nCu'].sum():,.2f} %")