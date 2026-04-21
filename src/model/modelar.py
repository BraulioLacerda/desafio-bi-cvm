import os
import json
import warnings
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

CAMINHO_DADOS = "data/processed/dataset_processado.csv"
PASTA_SAIDA = "data/processed"

QUANTIL_INFERIOR = 0.01
QUANTIL_SUPERIOR = 0.99


def transformar_target(y):
    return np.sign(y) * np.log1p(np.abs(y))


def desfazer_transformacao(y):
    return np.sign(y) * np.expm1(np.abs(y))


def carregar_dados():
    df = pd.read_csv(CAMINHO_DADOS)
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"], errors="coerce")
    return df


def preparar_dados(df):
    df = df.dropna(subset=["fluxo_futuro_pct", "DT_COMPTC"]).copy()
    df = df.sort_values("DT_COMPTC").copy()

    y_original = df["fluxo_futuro_pct"].copy()

    limite_inf = y_original.quantile(QUANTIL_INFERIOR)
    limite_sup = y_original.quantile(QUANTIL_SUPERIOR)
    mascara = (y_original >= limite_inf) & (y_original <= limite_sup)

    df = df.loc[mascara].copy()
    y_original = df["fluxo_futuro_pct"].copy()
    y = transformar_target(y_original)

    colunas_remover = [
        "fluxo_futuro_pct",
        "fluxo_futuro",
        "pl_defasado_1d",
        "DT_COMPTC",
        "CNPJ_FUNDO_CLASSE",
        "CNPJ_FUNDO_CLASSE_norm",
        "ID_SUBCLASSE"
    ]

    X = df.drop(columns=[c for c in colunas_remover if c in df.columns], errors="ignore").copy()
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    if X.shape[1] == 0:
        raise ValueError("Nenhuma coluna numérica disponível para o modelo.")

    return X, y, y_original, df, limite_inf, limite_sup


def split_temporal(df, X, y, y_original):
    df = df.sort_values("DT_COMPTC").copy()

    X = X.loc[df.index]
    y = y.loc[df.index]
    y_original = y_original.loc[df.index]

    corte = int(len(df) * 0.8)

    X_train = X.iloc[:corte].copy()
    X_test = X.iloc[corte:].copy()

    y_train = y.iloc[:corte].copy()
    y_test = y.iloc[corte:].copy()

    y_test_original = y_original.iloc[corte:].copy()
    df_test = df.iloc[corte:].copy()

    return X_train, X_test, y_train, y_test, y_test_original, df_test


def treinar_modelo(X_train, y_train):
    modelo = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    return modelo


def avaliar_modelo(modelo, X_test, y_test_original):
    y_pred_transformado = modelo.predict(X_test)
    y_pred_original = desfazer_transformacao(y_pred_transformado)

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)

    return y_pred_original, rmse, mae


def salvar_resultados(
    df_test,
    y_test_original,
    y_pred_original,
    modelo,
    X_train,
    rmse,
    mae,
    limite_inf,
    limite_sup
):
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    resultado = df_test.copy()
    resultado["valor_real"] = y_test_original.values
    resultado["predicao"] = y_pred_original
    resultado["erro_absoluto"] = np.abs(resultado["valor_real"] - resultado["predicao"])

    resultado.to_csv(os.path.join(PASTA_SAIDA, "predicoes_modelo.csv"), index=False)

    try:
        resultado.to_excel(os.path.join(PASTA_SAIDA, "predicoes_modelo.xlsx"), index=False)
    except Exception:
        pass

    importancias = pd.DataFrame({
        "variavel": X_train.columns,
        "importancia": modelo.feature_importances_
    }).sort_values("importancia", ascending=False)

    importancias.to_csv(os.path.join(PASTA_SAIDA, "importancia_variaveis.csv"), index=False)

    try:
        importancias.to_excel(os.path.join(PASTA_SAIDA, "importancia_variaveis.xlsx"), index=False)
    except Exception:
        pass

    metricas = {
        "modelo": "XGBRegressor",
        "rmse": float(rmse),
        "mae": float(mae),
        "quantil_inferior_outlier": float(QUANTIL_INFERIOR),
        "quantil_superior_outlier": float(QUANTIL_SUPERIOR),
        "limite_inferior_target": float(limite_inf),
        "limite_superior_target": float(limite_sup),
        "n_linhas_teste": int(len(df_test)),
        "n_features": int(X_train.shape[1]),
    }

    with open(os.path.join(PASTA_SAIDA, "metricas_modelo.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)


def main():
    df = carregar_dados()
    X, y, y_original, df, limite_inf, limite_sup = preparar_dados(df)
    X_train, X_test, y_train, y_test, y_test_original, df_test = split_temporal(df, X, y, y_original)

    modelo = treinar_modelo(X_train, y_train)
    y_pred_original, rmse, mae = avaliar_modelo(modelo, X_test, y_test_original)

    salvar_resultados(
        df_test=df_test,
        y_test_original=y_test_original,
        y_pred_original=y_pred_original,
        modelo=modelo,
        X_train=X_train,
        rmse=rmse,
        mae=mae,
        limite_inf=limite_inf,
        limite_sup=limite_sup
    )

    print(f"RMSE: {rmse:.8f}")
    print(f"MAE: {mae:.8f}")


if __name__ == "__main__":
    main()