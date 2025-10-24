import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# --- Configuração inicial do app ---
st.set_page_config(page_title="Machine Learning na Saúde", layout="wide")

st.title("🩺 Machine Learning Aplicado à Saúde")
st.write("Aplicação interativa demonstrando técnicas de aprendizado supervisionado e não supervisionado em datasets da área da saúde.")

# --- Escolha do dataset ---
dataset_choice = st.sidebar.selectbox("Escolha o dataset:", [
    "Diabetes (Regressão)",
    "Câncer de Mama (Classificação)",
    "Enviar CSV Próprio"
])

# --- Carregar dataset selecionado ---
if dataset_choice == "Diabetes (Regressão)":
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    problem_type = "regression"

elif dataset_choice == "Câncer de Mama (Classificação)":
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    problem_type = "classification"

else:
    uploaded = st.sidebar.file_uploader("Envie seu arquivo CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Prévia dos dados enviados:")
        st.dataframe(df.head())
        y_column = st.sidebar.selectbox("Escolha a coluna alvo (target):", df.columns)
        X = df.drop(columns=[y_column])
        y = df[y_column]
        problem_type = st.sidebar.radio("Tipo de problema:", ["regression", "classification"])
    else:
        st.warning("Envie um arquivo CSV para continuar.")
        st.stop()

# --- Análise Exploratória (EDA) ---
st.subheader("🔍 Análise Exploratória de Dados (EDA)")
st.write("Visualização inicial dos dados selecionados.")

col1, col2 = st.columns(2)
with col1:
    st.write("**Dimensões:**", X.shape)
    st.write("**Colunas:**", list(X.columns))
with col2:
    st.write("**Target:**", y.name)
    st.write("**Valores únicos do target:**", y.unique()[:10])

st.write("### Estatísticas descritivas")
st.dataframe(X.describe())

# --- Correlação ---
st.write("### Mapa de Correlação")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), cmap="Blues", ax=ax)
st.pyplot(fig)

# --- Separação treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Escalonamento ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Aprendizado Supervisionado ---
st.subheader("🤖 Aprendizado Supervisionado")

if problem_type == "regression":
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    st.write("**Modelo:** Regressão Linear")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

else:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    st.write("**Modelo:** Regressão Logística")
    st.write(f"**Acurácia:** {acc:.2f}")
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, preds))

# --- Salvar modelo ---
if st.button("💾 Baixar modelo treinado"):
    joblib.dump(model, "modelo_treinado.joblib")
    st.success("Modelo salvo como 'modelo_treinado.joblib'")

# --- Aprendizado Não Supervisionado ---
st.subheader("🧩 Aprendizado Não Supervisionado (KMeans + PCA)")
num_clusters = st.slider("Escolha o número de clusters (K):", 2, 10, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_train_scaled)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax2)
st.pyplot(fig2)

st.success("Análise concluída com sucesso! 🚀")

