"""
SCIC - Sistema de Comunicação Interplanetária da Colônia
Aurora Siger | Atividade Integradora
Protótipo acadêmico: dados simulados, análise numérica, regressão,
heap de alertas, trie de busca, eletricidade básica e discussão de gestão inteligente.
"""

from pathlib import Path
import heapq
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "dados" / "dados_aurora_siger.csv"

def carregar_dados():
    df = pd.read_csv(DATA)
    return df

def calcular_indicadores(df):
    erro = df["latencia_observada_ms"] - df["latencia_estimada_ms"]
    df = df.copy()
    df["erro_absoluto_ms"] = erro.abs()
    df["erro_relativo_pct"] = np.where(
        df["latencia_estimada_ms"] != 0,
        (df["erro_absoluto_ms"] / df["latencia_estimada_ms"]) * 100,
        0
    )
    df["potencia_calculada_W"] = df["tensao_V"] * df["corrente_A"]
    return df

def metricas_regressao(y_true, y_pred, n_params):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    rss = float(np.sum((np.asarray(y_true) - np.asarray(y_pred))**2))
    sigma2 = max(rss/n, 1e-12)
    aic = n * math.log(sigma2) + 2*n_params
    bic = n * math.log(sigma2) + n_params * math.log(n)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "AIC": aic, "BIC": bic}

def treinar_modelo(df):
    # Previsão simples da latência observada.
    features = ["latencia_estimada_ms", "tensao_V", "corrente_A", "prioridade"]
    X = df[features]
    y = df["latencia_observada_ms"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = metricas_regressao(y_test, pred, len(features)+1)
    return model, metrics, X_test, y_test, pred

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False
        self.records = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, key, record):
        node = self.root
        for char in key.lower():
            node = node.children.setdefault(char, TrieNode())
        node.end = True
        node.records.append(record)

    def _collect(self, node, output):
        if node.end:
            output.extend(node.records)
        for child in node.children.values():
            self._collect(child, output)

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        result=[]
        self._collect(node,result)
        return result

def construir_trie(df):
    trie = Trie()
    for _, row in df.iterrows():
        record = {
            "codigo": row["codigo_modulo"],
            "nome": row["nome_modulo"],
            "tipo": row["tipo_modulo"],
            "alerta": row["mensagem_alerta"]
        }
        # Permite busca por nome, tipo e código.
        for key in [str(row["nome_modulo"]), str(row["tipo_modulo"]), str(row["codigo_modulo"])]:
            trie.insert(key, record)
    return trie

def construir_heap(df):
    # heapq é uma fila de prioridade. Menor valor sai primeiro.
    heap=[]
    for _, row in df.iterrows():
        # prioridade 5 é mais crítica; transformamos em negativo.
        score = -(int(row["prioridade"]) * 100 + int(min(row["erro_absoluto_ms"], 99)))
        heapq.heappush(heap, (score, row["codigo_modulo"], row["mensagem_alerta"]))
    return heap

def analisar_alertas(df):
    heap = construir_heap(df)
    print("\n=== ALERTAS PRIORITÁRIOS (HEAP) ===")
    for i in range(min(5, len(heap))):
        score, code, msg = heapq.heappop(heap)
        print(f"{i+1}. {code} | prioridade calculada={-score} | {msg}")

def mostrar_eletricidade(df):
    print("\n=== ELETRICIDADE BÁSICA ===")
    cols=["codigo_modulo","tensao_V","corrente_A","potencia_W","potencia_calculada_W"]
    print(df[cols].head(8).to_string(index=False))
    print("\nFórmula: P = V × I")

def gerar_graficos(df):
    import matplotlib.pyplot as plt
    out=BASE/"graficos"
    out.mkdir(exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.scatter(df["latencia_estimada_ms"], df["latencia_observada_ms"])
    plt.xlabel("Latência estimada (ms)")
    plt.ylabel("Latência observada (ms)")
    plt.title("Latência estimada x observada")
    plt.tight_layout(); plt.savefig(out/"latencia_estim_vs_obs.png",dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    df.groupby("tipo_modulo")["erro_absoluto_ms"].mean().sort_values().plot(kind="bar")
    plt.ylabel("Erro absoluto médio (ms)")
    plt.title("Erro absoluto médio por tipo de módulo")
    plt.tight_layout(); plt.savefig(out/"erro_por_tipo.png",dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    df["prioridade"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Prioridade")
    plt.ylabel("Quantidade de registros")
    plt.title("Distribuição das prioridades")
    plt.tight_layout(); plt.savefig(out/"distribuicao_prioridades.png",dpi=150); plt.close()

def main():
    df=carregar_dados()
    df=calcular_indicadores(df)
    print("=== SCIC | AURORA SIGER ===")
    print(f"Registros carregados: {len(df)}")
    print("\nResumo:")
    print(df[["latencia_estimada_ms","latencia_observada_ms","erro_absoluto_ms","erro_relativo_pct"]].describe().round(2))

    model, metrics, X_test, y_test, pred = treinar_modelo(df)
    print("\n=== MODELO DE PREVISÃO ===")
    print("Regressão linear para estimar latência observada.")
    for k,v in metrics.items(): print(f"{k}: {v:.4f}")
    print("\nCoeficientes:")
    for name,coef in zip(["latencia_estimada_ms","tensao_V","corrente_A","prioridade"],model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercepto: {model.intercept_:.4f}")

    analisar_alertas(df)
    trie=construir_trie(df)
    prefixo=input("\nDigite um prefixo para busca (ex.: Com, Ant, AGR): ").strip() or "Com"
    resultados=trie.search_prefix(prefixo)
    print(f"\n=== BUSCA POR PREFIXO: '{prefixo}' ===")
    if resultados:
        for r in resultados[:10]: print(r)
    else: print("Nenhum registro encontrado.")

    mostrar_eletricidade(df)
    gerar_graficos(df)
    print("\nGráficos salvos em /graficos.")
    print("\nConclusão: o protótipo organiza os dados, mede erros, prevê latência, prioriza alertas e acelera buscas, apoiando decisões da colônia.")

if __name__ == "__main__":
    main()
