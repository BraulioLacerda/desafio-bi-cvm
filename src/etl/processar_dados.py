import os
import glob
import pandas as pd
import numpy as np


PASTA_BRUTA_INF = os.path.join("data", "raw", "inf_diario")
PASTA_BRUTA_CAD = os.path.join("data", "raw", "cadastro")
PASTA_PROCESSADO = os.path.join("data", "processed")

HORIZONTE_ALVO = 21
TAMANHO_CHUNK = 200_000


def normalizar_cnpj(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("nan", np.nan)
        .replace("None", np.nan)
        .replace("", np.nan)
    )


def converter_numerico_serie(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return serie

    return pd.to_numeric(
        serie.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace(["nan", "None", ""], np.nan),
        errors="coerce"
    )


def carregar_inf_diario(caminho_dados: str) -> pd.DataFrame:
    print(f"\n[1/6] Buscando arquivos CSV em: {caminho_dados}")

    arquivos_csv = sorted(glob.glob(os.path.join(caminho_dados, "*.csv")))

    if not arquivos_csv:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {caminho_dados}.")

    print(f"Total de arquivos encontrados: {len(arquivos_csv)}")

    colunas_desejadas = {
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
    }

    quadros = []

    for i, arquivo in enumerate(arquivos_csv, start=1):
        nome = os.path.basename(arquivo)
        print(f"  -> Lendo arquivo {i}/{len(arquivos_csv)}: {nome}")

        leitor = pd.read_csv(
            arquivo,
            sep=";",
            low_memory=False,
            chunksize=TAMANHO_CHUNK
        )

        for chunk in leitor:
            cols_presentes = [c for c in chunk.columns if c in colunas_desejadas]
            chunk = chunk[cols_presentes].copy()
            quadros.append(chunk)

    combinado = pd.concat(quadros, ignore_index=True)
    print(f"Leitura do informe diário concluída. Shape: {combinado.shape}")
    return combinado


def _carregar_cadastro_registro(caminho_dados: str) -> pd.DataFrame | None:
    caminho_fundo = os.path.join(caminho_dados, "registro_fundo.csv")
    caminho_classe = os.path.join(caminho_dados, "registro_classe.csv")
    caminho_subclasse = os.path.join(caminho_dados, "registro_subclasse.csv")

    if not all(os.path.exists(arq) for arq in [caminho_fundo, caminho_classe, caminho_subclasse]):
        return None

    fundo = pd.read_csv(caminho_fundo, sep=";", low_memory=False, encoding="latin-1")
    classe = pd.read_csv(caminho_classe, sep=";", low_memory=False, encoding="latin-1")
    subclasse = pd.read_csv(caminho_subclasse, sep=";", low_memory=False, encoding="latin-1")

    cadastro = fundo.merge(classe, how="left", on="ID_Registro_Fundo")
    cadastro = cadastro.merge(subclasse, how="left", on="ID_Registro_Classe")

    if "CNPJ_Classe" in cadastro.columns:
        cadastro["CNPJ_Classe_norm"] = normalizar_cnpj(cadastro["CNPJ_Classe"])

    if "ID_Subclasse" in cadastro.columns:
        cadastro["ID_Subclasse"] = cadastro["ID_Subclasse"].astype(str)

    cadastro["origem_cadastro"] = "registro_fundo_classe"
    return cadastro


def _carregar_cadastro_cad_fi(caminho_dados: str) -> pd.DataFrame | None:
    caminho_cad_fi = os.path.join(caminho_dados, "cad_fi.csv")

    if not os.path.exists(caminho_cad_fi):
        return None

    cadastro = pd.read_csv(caminho_cad_fi, sep=";", low_memory=False, encoding="latin-1")

    if "CNPJ_FUNDO" in cadastro.columns:
        cadastro["CNPJ_FUNDO_norm"] = normalizar_cnpj(cadastro["CNPJ_FUNDO"])

    cadastro["origem_cadastro"] = "cad_fi"
    return cadastro


def carregar_cadastro_fundos(caminho_dados: str) -> pd.DataFrame:
    print(f"\n[2/6] Carregando cadastro em: {caminho_dados}")

    cadastro = _carregar_cadastro_registro(caminho_dados)
    if cadastro is not None:
        print(f"Cadastro carregado via registro_fundo_classe. Shape: {cadastro.shape}")
        return cadastro

    cadastro = _carregar_cadastro_cad_fi(caminho_dados)
    if cadastro is not None:
        print(f"Cadastro carregado via cad_fi.csv. Shape: {cadastro.shape}")
        return cadastro

    raise FileNotFoundError(
        "Não encontrei nem os arquivos extraídos do registro_fundo_classe.zip "
        "nem o cad_fi.csv dentro de data/raw/cadastro."
    )


def filtrar_fundos_acoes(inf_diario: pd.DataFrame, cadastro: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/6] Filtrando fundos de ações...")

    inf_diario = inf_diario.copy()
    inf_diario["CNPJ_FUNDO_CLASSE_norm"] = normalizar_cnpj(inf_diario["CNPJ_FUNDO_CLASSE"])
    inf_diario["ID_SUBCLASSE"] = inf_diario["ID_SUBCLASSE"].astype(str)

    origem = cadastro.get("origem_cadastro")
    if isinstance(origem, pd.Series):
        origem = origem.iloc[0]

    # CENÁRIO 1: estrutura antiga / mais completa

    if origem == "registro_fundo_classe" and "Classificacao_Anbima" in cadastro.columns:
        fundos_acoes = cadastro[
            cadastro["Classificacao_Anbima"].astype(str).str.contains("Ações", case=False, na=False)
        ].copy()

        print(f"Fundos/classes de ações encontrados no cadastro: {len(fundos_acoes):,}")

        if "CNPJ_Classe_norm" in fundos_acoes.columns:
            filtrado_cnpj = inf_diario.merge(
                fundos_acoes[["CNPJ_Classe_norm"]].dropna().drop_duplicates(),
                how="inner",
                left_on="CNPJ_FUNDO_CLASSE_norm",
                right_on="CNPJ_Classe_norm"
            )

            print(f"Linhas após filtro por CNPJ da classe: {len(filtrado_cnpj):,}")

            if len(filtrado_cnpj) > 0:
                return filtrado_cnpj.drop(columns=["CNPJ_Classe_norm"], errors="ignore")

        if "ID_Subclasse" in fundos_acoes.columns:
            filtrado_sub = inf_diario.merge(
                fundos_acoes[["ID_Subclasse"]].dropna().drop_duplicates(),
                how="inner",
                left_on="ID_SUBCLASSE",
                right_on="ID_Subclasse"
            )

            print(f"Linhas após filtro por ID_Subclasse: {len(filtrado_sub):,}")

            if len(filtrado_sub) > 0:
                return filtrado_sub.drop(columns=["ID_Subclasse"], errors="ignore")


    # CENÁRIO 2: cadastro

    print("Cadastro via cad_fi.csv ou sem chave confiável de classe. Usando fallback por TP_FUNDO_CLASSE.")

    if "TP_FUNDO_CLASSE" not in inf_diario.columns:
        raise KeyError("Coluna 'TP_FUNDO_CLASSE' não encontrada no informe diário.")

    fallback = inf_diario[
        inf_diario["TP_FUNDO_CLASSE"].astype(str).str.contains("Ações|Acoes", case=False, na=False)
    ].copy()

    print(f"Linhas após fallback por TP_FUNDO_CLASSE: {len(fallback):,}")

    if len(fallback) == 0:
        raise KeyError(
            "Não consegui identificar fundos de ações nem via cadastro nem via TP_FUNDO_CLASSE."
        )

    return fallback


def gerar_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Gerando características...")

    coluna_fundo = "CNPJ_FUNDO_CLASSE_norm"

    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"], errors="coerce")

    col_numericas = ["VL_TOTAL", "VL_QUOTA", "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "NR_COTST"]
    for col in col_numericas:
        if col in df.columns:
            df[col] = converter_numerico_serie(df[col])

    df = df.dropna(subset=["DT_COMPTC", coluna_fundo]).copy()
    df = df.sort_values([coluna_fundo, "DT_COMPTC"]).copy()

    # Fluxo líquido diário
    df["fluxo_liquido"] = df["CAPTC_DIA"] - df["RESG_DIA"]

    # Retorno diário
    df["retorno_diario"] = df.groupby(coluna_fundo)["VL_QUOTA"].pct_change()

    # Retornos acumulados
    for janela in [21, 63, 126]:
        df[f"ret_{janela}d"] = df.groupby(coluna_fundo)["VL_QUOTA"].pct_change(periods=janela)

    # Volatilidades
    for janela in [21, 63]:
        df[f"vol_{janela}d"] = (
            df.groupby(coluna_fundo)["retorno_diario"]
            .rolling(janela)
            .std()
            .reset_index(level=0, drop=True)
        )

    # Drawdown
    for janela in [63, 126]:
        max_rolado = (
            df.groupby(coluna_fundo)["VL_QUOTA"]
            .rolling(janela)
            .max()
            .reset_index(level=0, drop=True)
        )
        df[f"drawdown_{janela}d"] = (df["VL_QUOTA"] - max_rolado) / max_rolado

    # Z-score do retorno de 21 dias
    media_ret_21 = df.groupby(coluna_fundo)["ret_21d"].transform("mean")
    desvio_ret_21 = df.groupby(coluna_fundo)["ret_21d"].transform("std")
    df["zscore_ret_21d"] = (df["ret_21d"] - media_ret_21) / desvio_ret_21

    # Log do PL
    df["log_vl_pl"] = np.log(df["VL_PATRIM_LIQ"].replace(0, np.nan))

    # Variação de PL sem fluxo
    df["delta_pl"] = df.groupby(coluna_fundo)["VL_PATRIM_LIQ"].diff()
    df["var_pl_sem_fluxo"] = df["delta_pl"] - df["fluxo_liquido"]

    # Idade
    primeira_data = df.groupby(coluna_fundo)["DT_COMPTC"].transform("min")
    df["idade_fundo_dias"] = (df["DT_COMPTC"] - primeira_data).dt.days

    # Sazonalidade final
    df["fim_mes"] = df["DT_COMPTC"].dt.is_month_end.astype(int)
    df["fim_trimestre"] = df["DT_COMPTC"].dt.is_quarter_end.astype(int)

    print(f"Características geradas. Shape: {df.shape}")
    return df


def calcular_alvo(df: pd.DataFrame, horizonte: int = 21) -> pd.DataFrame:
    print("\n[5/6] Calculando alvo...")

    coluna_fundo = "CNPJ_FUNDO_CLASSE_norm"
    df = df.sort_values([coluna_fundo, "DT_COMPTC"]).copy()

    df["fluxo_futuro"] = (
        df.groupby(coluna_fundo)["fluxo_liquido"]
        .transform(lambda x: x.shift(-1).rolling(window=horizonte, min_periods=horizonte).sum())
    )

    df["pl_defasado_1d"] = df.groupby(coluna_fundo)["VL_PATRIM_LIQ"].shift(1)
    df["fluxo_futuro_pct"] = df["fluxo_futuro"] / df["pl_defasado_1d"]

    print("Alvo calculado.")
    return df


def executar_preparacao():
    print("INÍCIO DO PROCESSAMENTO")
    os.makedirs(PASTA_PROCESSADO, exist_ok=True)

    inf_diario = carregar_inf_diario(PASTA_BRUTA_INF)
    cadastro = carregar_cadastro_fundos(PASTA_BRUTA_CAD)
    inf_acoes = filtrar_fundos_acoes(inf_diario, cadastro)
    df_carac = gerar_caracteristicas(inf_acoes)
    df_final = calcular_alvo(df_carac, horizonte=HORIZONTE_ALVO)

    print("\n[6/6] Limpando e salvando saída...")
    df_final = df_final.dropna(subset=["fluxo_futuro_pct"]).copy()
    df_final = df_final.sort_values(["CNPJ_FUNDO_CLASSE_norm", "DT_COMPTC"]).copy()

    caminho_saida_csv = os.path.join(PASTA_PROCESSADO, "dataset_processado.csv")
    caminho_saida_xlsx = os.path.join(PASTA_PROCESSADO, "dataset_processado.xlsx")
    caminho_dict = os.path.join(PASTA_PROCESSADO, "dicionario_variaveis.txt")

    df_final.to_csv(caminho_saida_csv, index=False)
    df_final.to_excel(caminho_saida_xlsx, index=False)

    with open(caminho_dict, "w", encoding="utf-8") as f:
        f.write("\n".join(df_final.columns))

 
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"CSV salvo em: {caminho_saida_csv}")
    print(f"Excel salvo em: {caminho_saida_xlsx}")
    print(f"Dicionário salvo em: {caminho_dict}")
    print(f"Shape final: {df_final.shape}")
  


if __name__ == "__main__":
    executar_preparacao()