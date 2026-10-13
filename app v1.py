
import streamlit as st
import pandas as pd
from pathlib import Path

# CONFIGURAÇÃO
st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# CARREGAMENTO
@st.cache_data
def carregar_dados():
    caminho = Path(__file__).parent / "dashboard_churn.csv"
    return pd.read_csv(caminho)

df = carregar_dados()

# TÍTULO
st.title("Churn Analytics | Retenção de Clientes")

st.write(
    "Análise de modelos de Machine Learning "
    "e avaliação financeira de campanhas de retenção."
)

st.divider()

# INDICADORES
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de clientes", len(df))

with col2:
    st.metric(
        "Clientes com churn",
        int(df["Churn_Real"].sum())
    )

with col3:
    st.metric(
        "Mensalidade média",
        f"R$ {df['MonthlyCharges'].mean():.2f}"
    )

# TABELA
st.subheader("Dados de validação")

st.dataframe(
    df.head(20),
    use_container_width=True
)
