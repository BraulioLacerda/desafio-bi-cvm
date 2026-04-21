import os
import datetime
import requests
import zipfile
import io

# Diretórios de saída
RAW_DIR = os.path.join("data", "raw")
INF_DIR = os.path.join(RAW_DIR, "inf_diario")
CAD_DIR = os.path.join(RAW_DIR, "cadastro")

os.makedirs(INF_DIR, exist_ok=True)
os.makedirs(CAD_DIR, exist_ok=True)

def download_and_extract_zip(url, out_dir):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(out_dir)
    print(f"Baixado e extraído: {os.path.basename(url)}")

def gerar_intervalo_meses(n_meses: int):
    hoje = datetime.date.today()
    for i in range(n_meses):
        ano = hoje.year
        mes = hoje.month - i
        while mes <= 0:
            mes += 12
            ano -= 1
        yield f"{ano}{mes:02d}"

def baixar_informes_diarios(n_meses=12):
    base_url = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
    for ym in gerar_intervalo_meses(n_meses):
        file_name = f"inf_diario_fi_{ym}.zip"
        url = f"{base_url}/{file_name}"
        try:
            download_and_extract_zip(url, INF_DIR)
        except Exception as e:
            print(f"Erro ao baixar {file_name}: {e}")

def baixar_cadastro_fundos():
    url = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip"
    download_and_extract_zip(url, CAD_DIR)

if __name__ == "__main__":
    baixar_informes_diarios(n_meses=12)
    baixar_cadastro_fundos()