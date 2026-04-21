import os
import pandas as pd
import matplotlib.pyplot as plt


CAMINHO_DATASET = "data/processed/dataset_processado.csv"
CAMINHO_PREDICOES = "data/processed/predicoes_modelo.csv"
CAMINHO_IMPORTANCIAS = "data/processed/importancia_variaveis.csv"
PASTA_SAIDA = "reports/graficos"


def garantir_pasta():
    os.makedirs(PASTA_SAIDA, exist_ok=True)


def carregar_arquivos():
    df = pd.read_csv(CAMINHO_DATASET)
    pred = pd.read_csv(CAMINHO_PREDICOES)
    imp = pd.read_csv(CAMINHO_IMPORTANCIAS)

    if "DT_COMPTC" in df.columns:
        df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"], errors="coerce")

    if "DT_COMPTC" in pred.columns:
        pred["DT_COMPTC"] = pd.to_datetime(pred["DT_COMPTC"], errors="coerce")

    return df, pred, imp


def grafico_distribuicao_target(df):
    base = df["fluxo_futuro_pct"].dropna().copy()

    if len(base) == 0:
        return

    lower = base.quantile(0.01)
    upper = base.quantile(0.99)
    base = base[(base >= lower) & (base <= upper)]

    plt.figure(figsize=(8, 5))
    plt.hist(base, bins=40)
    plt.title("Distribuição do fluxo futuro (%)")
    plt.xlabel("Fluxo futuro (%)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "01_distribuicao_target.png"))
    plt.close()


def grafico_real_vs_previsto(pred):
    base = pred[["valor_real", "predicao"]].dropna().copy()

    if len(base) == 0:
        return

    lower = base["valor_real"].quantile(0.01)
    upper = base["valor_real"].quantile(0.99)

    base = base[
        (base["valor_real"] >= lower) &
        (base["valor_real"] <= upper)
    ].copy()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        base["valor_real"],
        base["predicao"],
        alpha=0.3
    )

    minimo = min(base["valor_real"].min(), base["predicao"].min())
    maximo = max(base["valor_real"].max(), base["predicao"].max())

    plt.plot([minimo, maximo], [minimo, maximo], linestyle="--")

    plt.title("Valor real vs previsão")
    plt.xlabel("Valor real")
    plt.ylabel("Previsão")
    plt.xlim(minimo, maximo)
    plt.ylim(minimo, maximo)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "02_real_vs_previsto.png"))
    plt.close()


def grafico_distribuicao_erros(pred):
    base = pred["erro_absoluto"].dropna().copy()

    if len(base) == 0:
        return

    upper = base.quantile(0.99)
    base = base[base <= upper]

    plt.figure(figsize=(8, 5))
    plt.hist(base, bins=40)
    plt.title("Distribuição do erro absoluto")
    plt.xlabel("Erro absoluto")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "03_distribuicao_erro.png"))
    plt.close()


def grafico_top_importancias(imp, top_n=10):
    base = imp.head(top_n).sort_values("importancia", ascending=True).copy()

    if len(base) == 0:
        return

    plt.figure(figsize=(9, 6))
    plt.barh(base["variavel"], base["importancia"])
    plt.title(f"Top {top_n} variáveis mais importantes")
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "04_top_importancias.png"))
    plt.close()


def grafico_fluxo_por_quantis_retorno(df):
    if "ret_63d" not in df.columns or "fluxo_futuro_pct" not in df.columns:
        return

    base = df[["ret_63d", "fluxo_futuro_pct"]].dropna().copy()

    if len(base) == 0:
        return

    base["quantil_retorno"] = pd.qcut(
        base["ret_63d"],
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop"
    )

    resumo = (
        base.groupby("quantil_retorno", observed=False)["fluxo_futuro_pct"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.bar(resumo["quantil_retorno"], resumo["fluxo_futuro_pct"])
    plt.title("Fluxo futuro médio por quantil de retorno (63d)")
    plt.xlabel("Quantil de retorno")
    plt.ylabel("Fluxo futuro médio")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "05_fluxo_por_quantil_retorno.png"))
    plt.close()


def grafico_fluxo_por_quantis_volatilidade(df):
    if "vol_63d" not in df.columns or "fluxo_futuro_pct" not in df.columns:
        return

    base = df[["vol_63d", "fluxo_futuro_pct"]].dropna().copy()

    if len(base) == 0:
        return

    lower = base["fluxo_futuro_pct"].quantile(0.01)
    upper = base["fluxo_futuro_pct"].quantile(0.99)
    base = base[
        (base["fluxo_futuro_pct"] >= lower) &
        (base["fluxo_futuro_pct"] <= upper)
    ].copy()

    base["quantil_vol"] = pd.qcut(
        base["vol_63d"],
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop"
    )

    resumo = (
        base.groupby("quantil_vol", observed=False)["fluxo_futuro_pct"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.bar(resumo["quantil_vol"], resumo["fluxo_futuro_pct"])
    plt.title("Fluxo futuro médio por quantil de volatilidade (63d)")
    plt.xlabel("Quantil de volatilidade")
    plt.ylabel("Fluxo futuro médio")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "06_fluxo_por_quantil_volatilidade.png"))
    plt.close()


def grafico_serie_temporal_predicoes(pred):
    if "DT_COMPTC" not in pred.columns:
        return

    base = pred[["DT_COMPTC", "valor_real", "predicao"]].dropna().copy()

    if len(base) == 0:
        return
    base = (
        base.groupby("DT_COMPTC")[["valor_real", "predicao"]]
        .mean()
        .reset_index()
    )

    base = base.sort_values("DT_COMPTC").copy()

    # suavização leve
    base["real_suavizado"] = base["valor_real"].rolling(3).mean()
    base["prev_suavizado"] = base["predicao"].rolling(3).mean()

    plt.figure(figsize=(10, 5))

    plt.plot(base["DT_COMPTC"], base["real_suavizado"], label="Real (médio)")
    plt.plot(base["DT_COMPTC"], base["prev_suavizado"], label="Previsto (médio)")

    plt.title("Fluxo médio ao longo do tempo")
    plt.xlabel("Data")
    plt.ylabel("Fluxo futuro (%)")

    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/graficos/07_serie_temporal_real_previsto.png")
    plt.close()

def main():
    garantir_pasta()
    df, pred, imp = carregar_arquivos()

    grafico_distribuicao_target(df)
    grafico_real_vs_previsto(pred)
    grafico_distribuicao_erros(pred)
    grafico_top_importancias(imp, top_n=10)
    grafico_fluxo_por_quantis_retorno(df)
    grafico_fluxo_por_quantis_volatilidade(df)
    grafico_serie_temporal_predicoes(pred)

    print(f"Gráficos gerados com sucesso em: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()