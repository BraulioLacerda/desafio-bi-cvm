import os
import pandas as pd
import numpy as np


PASTA_PROCESSADO = os.path.join("data", "processed")
PASTA_SAIDA = os.path.join("docs", "validacao_manual")

os.makedirs(PASTA_SAIDA, exist_ok=True)

ARQUIVO_DATASET = os.path.join(PASTA_PROCESSADO, "dataset_processado.csv")

# CONFIGURAÇÕES
N_FUNDOS = 2
N_DATAS = 3
SEED = 42

ARQUIVO_INF_DIARIO_REFERENCIA = "inf_diario_fi_202506.csv"

COLUNAS_VALIDACAO = [
    "TP_FUNDO_CLASSE",
    "CNPJ_FUNDO_CLASSE",
    "ID_SUBCLASSE",
    "DT_COMPTC",
    "VL_TOTAL",
    "VL_QUOTA",
    "VL_PATRIM_LIQ",
    "CAPTC_DIA",
    "RESG_DIA",
    "NR_COTST",
]


def normalizar_cnpj(serie):
    return (
        serie.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


def carregar_dataset():
    df = pd.read_csv(ARQUIVO_DATASET, low_memory=False)
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"], errors="coerce")

    if "CNPJ_FUNDO_CLASSE_norm" in df.columns:
        df["CNPJ_FUNDO_CLASSE_norm"] = normalizar_cnpj(df["CNPJ_FUNDO_CLASSE_norm"])
    elif "CNPJ_FUNDO_CLASSE" in df.columns:
        df["CNPJ_FUNDO_CLASSE_norm"] = normalizar_cnpj(df["CNPJ_FUNDO_CLASSE"])
    else:
        raise KeyError(
            "O dataset processado não possui nem CNPJ_FUNDO_CLASSE_norm nem CNPJ_FUNDO_CLASSE."
        )

    return df


def escolher_fundos(df):
    np.random.seed(SEED)

    contagem = (
        df.groupby("CNPJ_FUNDO_CLASSE_norm")["DT_COMPTC"]
        .nunique()
        .sort_values(ascending=False)
    )

    elegiveis = contagem[contagem >= N_DATAS].index.tolist()

    if len(elegiveis) < N_FUNDOS:
        raise ValueError(
            f"Não há fundos suficientes com pelo menos {N_DATAS} datas válidas."
        )

    escolhidos = np.random.choice(elegiveis, size=N_FUNDOS, replace=False)
    return escolhidos


def filtrar_datas_por_arquivo_referencia(datas):
    if not ARQUIVO_INF_DIARIO_REFERENCIA:
        return datas

    nome = ARQUIVO_INF_DIARIO_REFERENCIA.replace(".csv", "")
    partes = nome.split("_")
    if len(partes) < 4:
        return datas

    ym = partes[-1]
    if len(ym) != 6 or not ym.isdigit():
        return datas

    ano = int(ym[:4])
    mes = int(ym[4:6])

    datas_filtradas = datas[(datas.dt.year == ano) & (datas.dt.month == mes)]

    if len(datas_filtradas) >= N_DATAS:
        return datas_filtradas

    return datas


def formatar_decimal(valor, casas):
    if pd.isna(valor) or valor == "":
        return ""
    return f"{float(valor):.{casas}f}"


def gerar_guia_validacao():
    print("================================")
    print("GERANDO GUIA DE VALIDAÇÃO")
    print("================================")
    print(f"Arquivo bruto de referência: {ARQUIVO_INF_DIARIO_REFERENCIA}")

    df = carregar_dataset()
    fundos = escolher_fundos(df)

    linhas = []

    for cnpj in fundos:
        base = df[df["CNPJ_FUNDO_CLASSE_norm"] == cnpj].copy()

        datas = (
            base["DT_COMPTC"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )

        datas = filtrar_datas_por_arquivo_referencia(datas)

        if len(datas) < N_DATAS:
            print(f"⚠ Fundo {cnpj} não possui datas suficientes após filtro do mês de referência.")
            continue

        datas_escolhidas = np.random.choice(datas, size=N_DATAS, replace=False)
        datas_escolhidas = sorted(pd.to_datetime(datas_escolhidas))

        for data in datas_escolhidas:
            linha = base[base["DT_COMPTC"] == data].iloc[0]

            registro = {}

            for col in COLUNAS_VALIDACAO:
                valor = linha[col] if col in linha.index else np.nan

                if col == "DT_COMPTC" and pd.notna(valor):
                    valor = pd.to_datetime(valor).strftime("%Y-%m-%d")

                registro[col] = valor

            linhas.append(registro)
            print(f"✔ {cnpj} - {data.date()}")

    if not linhas:
        raise ValueError(
            "Nenhuma linha foi selecionada para validação. "
            "Revise o ARQUIVO_INF_DIARIO_REFERENCIA ou a base processada."
        )

    df_out = pd.DataFrame(linhas)
    df_out = df_out.reindex(columns=COLUNAS_VALIDACAO)

    # FORMATAÇÃO
    if "VL_QUOTA" in df_out.columns:
        df_out["VL_QUOTA"] = df_out["VL_QUOTA"].apply(lambda x: formatar_decimal(x, 12))

    for col in ["VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "VL_TOTAL"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(lambda x: formatar_decimal(x, 2))

    if "NR_COTST" in df_out.columns:
        df_out["NR_COTST"] = (
            pd.to_numeric(df_out["NR_COTST"], errors="coerce")
            .fillna("")
            .apply(lambda x: "" if x == "" else str(int(float(x))))
        )

    caminho_csv = os.path.join(PASTA_SAIDA, "guia_validacao.csv")
    df_out.to_csv(caminho_csv, index=False, sep=";", encoding="utf-8-sig")

    print("\nArquivo salvo em:")
    print(caminho_csv)
    print("\nColunas exportadas:")
    print(";".join(COLUNAS_VALIDACAO))


if __name__ == "__main__":
    gerar_guia_validacao()