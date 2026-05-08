import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# LEITURA DO DATASET
# =========================

df = pd.read_csv("diabetes.csv")

# =========================
# VISUALIZAÇÃO DOS DADOS
# =========================

print("\nPrimeiras linhas do dataset:\n")
print(df.head())

print("\nInformações do dataset:\n")
print(df.info())

print("\nEstatísticas gerais:\n")
print(df.describe())

# =========================
# VERIFICAÇÃO DE VALORES NULOS
# =========================

print("\nValores nulos:\n")
print(df.isnull().sum())

# =========================
# GRÁFICO 1 - PACIENTES COM E SEM DIABETES
# =========================

plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df)

plt.title("Distribuição de Diabetes")
plt.xlabel("Resultado")
plt.ylabel("Quantidade")

plt.xticks([0,1], ["Não Diabético", "Diabético"])

plt.savefig("grafico_distribuicao.png")

# =========================
# GRÁFICO 2 - CORRELAÇÃO
# =========================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Mapa de Correlação")

plt.savefig("grafico_correlacao.png")

# =========================
# GRÁFICO 3 - GLICOSE X RESULTADO
# =========================

plt.figure(figsize=(8,5))

sns.boxplot(
    x='Outcome',
    y='Glucose',
    data=df
)

plt.title("Nível de Glicose por Resultado")
plt.xlabel("Resultado")
plt.ylabel("Glicose")

plt.xticks([0,1], ["Não Diabético", "Diabético"])

plt.savefig("grafico_glicose.png")

# =========================
# PREPARAÇÃO DOS DADOS
# =========================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Separação treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# CRIAÇÃO DO MODELO
# =========================

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Treinamento

modelo.fit(X_train, y_train)

# =========================
# PREVISÕES
# =========================

previsoes = modelo.predict(X_test)

# =========================
# RESULTADOS
# =========================

acuracia = accuracy_score(y_test, previsoes)

print("\nAcurácia do modelo:")
print(acuracia)

print("\nMatriz de Confusão:\n")
print(confusion_matrix(y_test, previsoes))

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, previsoes))

# =========================
# IMPORTÂNCIA DAS VARIÁVEIS
# =========================

importancias = pd.Series(
    modelo.feature_importances_,
    index=X.columns
)

plt.figure(figsize=(8,5))

importancias.sort_values().plot(kind='barh')

plt.title("Importância das Variáveis")

plt.xlabel("Importância")

plt.savefig("grafico_importancia.png")