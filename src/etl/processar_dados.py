import os
import glob
import pandas as pd
import numpy as np

def load_inf_diario(data_path):
    """Combina todos os CSVs de Informe Diário em um único DataFrame."""
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    frames = []
    for csv in csv_files:
        df = pd.read_csv(csv, sep=";", decimal=",")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined

def load_cadastro_fundos(data_path):
    """Lê os arquivos de cadastro (fundos, classes e subclasses) e junta num único DataFrame."""
    reg_fundo = pd.read_csv(os.path.join(data_path, "registro_fundo.csv"), sep=";", decimal=",")
    reg_classe = pd.read_csv(os.path.join(data_path, "registro_classe.csv"), sep=";", decimal=",")
    reg_subclasse = pd.read_csv(os.path.join(data_path, "registro_subclasse.csv"), sep=";", decimal=",")
    reg = reg_fundo.merge(reg_classe, how="left", on="ID_CLASSE")
    reg = reg.merge(reg_subclasse, how="left", on="ID_SUBCLASSE")
    return reg

def filter_fundos_acoes(inf_diario, cadastro):
    """Seleciona apenas os fundos com classe 'Ações'."""
    class_column = "DENOM_CLASSE" if "DENOM_CLASSE" in cadastro.columns else "CLASSE"
    fundos_acoes = cadastro[cadastro[class_column].str.contains("Ações", case=False, na=False)]
    inf_filtrado = inf_diario.merge(fundos_acoes[["CNPJ_FUNDO"]], how="inner", on="CNPJ_FUNDO")
    return inf_filtrado

def prepare_features(df):
    """Gera features de retorno, volatilidade, drawdown, z‑score, log do PL, variação de PL, idade do fundo e sazonalidade."""
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"])
    df.sort_values(["CNPJ_FUNDO", "DT_COMPTC"], inplace=True)

    df["fluxo_liquido"] = df["CAPTC_DIA"] - df["RESG_DIA"]
    df["retorno_diario"] = df.groupby("CNPJ_FUNDO")["VL_QUOTA"].pct_change()
    for window in [21, 63, 126]:
        df[f"ret_{window}d"] = df.groupby("CNPJ_FUNDO")["VL_QUOTA"].pct_change(periods=window)
    for window in [21, 63]:
        df[f"vol_{window}d"] = df.groupby("CNPJ_FUNDO")["retorno_diario"].rolling(window).std().reset_index(level=0, drop=True)
    for window in [63, 126]:
        rolling_max = df.groupby("CNPJ_FUNDO")["VL_QUOTA"].rolling(window).max().reset_index(level=0, drop=True)
        df[f"drawdown_{window}d"] = (df["VL_QUOTA"] - rolling_max) / rolling_max
    df["zscore_ret_21d"] = (df["ret_21d"] - df.groupby("CNPJ_FUNDO")["ret_21d"].transform("mean")) / df.groupby("CNPJ_FUNDO")["ret_21d"].transform("std")
    df["log_vl_pl"] = np.log(df["VL_PATRIM_LIQ"].replace(0, np.nan))
    df["delta_pl"] = df.groupby("CNPJ_FUNDO")["VL_PATRIM_LIQ"].diff()
    df["var_pl_sem_fluxo"] = df["delta_pl"] - df["fluxo_liquido"]
    first_dates = df.groupby("CNPJ_FUNDO")["DT_COMPTC"].transform("min")
    df["fund_age_days"] = (df["DT_COMPTC"] - first_dates).dt.days
    df["day_of_week"] = df["DT_COMPTC"].dt.weekday
    df["is_month_end"] = df["DT_COMPTC"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["DT_COMPTC"].dt.is_quarter_end.astype(int)
    return df

def compute_target(df, horizon=21):
    """Soma o fluxo líquido dos próximos horizon dias úteis e divide pelo PL defasado em 1 dia."""
    df = df.sort_values(["CNPJ_FUNDO", "DT_COMPTC"])
    df["flow_future"] = df.groupby("CNPJ_FUNDO")["fluxo_liquido"].apply(
        lambda x: x.rolling(window=horizon, min_periods=1).sum().shift(-horizon + 1)
    )
    df["pl_lag_1d"] = df.groupby("CNPJ_FUNDO")["VL_PATRIM_LIQ"].shift(1)
    df["flow_future_pct"] = df["flow_future"] / df["pl_lag_1d"]
    return df

def main():
    raw_inf_path = os.path.join("data", "raw", "inf_diario")
    raw_cad_path = os.path.join("data", "raw", "cadastro")
    processed_path = os.path.join("data", "processed")
    os.makedirs(processed_path, exist_ok=True)

    # Carrega e filtra
    inf_diario = load_inf_diario(raw_inf_path)
    cadastro = load_cadastro_fundos(raw_cad_path)
    inf_acoes = filter_fundos_acoes(inf_diario, cadastro)

    # Converte colunas numéricas com vírgula como decimal
    for col in ["VL_QUOTA", "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA"]:
        if inf_acoes[col].dtype == object:
            inf_acoes[col] = (inf_acoes[col]
                              .str.replace(".", "", regex=False)
                              .str.replace(",", ".", regex=False)
                              .astype(float))

    df_features = prepare_features(inf_acoes)
    df_final = compute_target(df_features, horizon=21)

    # Remove linhas sem target
    df_final = df_final.dropna(subset=["flow_future_pct"])

    # Salva dataset processado e dicionário de variáveis
    out_file = os.path.join(processed_path, "dataset_processed.csv")
    df_final.to_csv(out_file, index=False)
    with open(os.path.join(processed_path, "variable_dict.txt"), "w") as f:
        f.write("\n".join(df_final.columns))

    print(f"Processamento concluído. Arquivo salvo em {out_file}")

if __name__ == "__main__":
    main()
