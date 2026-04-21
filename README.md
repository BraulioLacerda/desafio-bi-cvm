# Desafio BI – CVM

## Objetivo

Este projeto tem como objetivo analisar os determinantes da captação líquida futura de fundos de investimento da classe Ações, utilizando dados públicos da CVM. A proposta combina tratamento de dados, modelagem estatística e validação, com foco tanto na capacidade preditiva quanto na interpretação econômica dos resultados.

---

## Definição do Problema

O problema foi estruturado como uma regressão, na qual a variável alvo corresponde ao fluxo percentual futuro dos fundos, definido como a soma dos fluxos líquidos entre T+1 e T+21 dias úteis, normalizada pelo patrimônio líquido defasado.

Essa abordagem permite comparar fundos de diferentes tamanhos e evita circularidade na construção da variável dependente.

---

## Estrutura do Projeto

```text
desafio-bi-cvm/
│
├── src/
│   ├── etl/
│   │   ├── baixar_dados.py
│   │   ├── processar_dados.py
│   │   └── validar_manual.py
│   │
│   └── model/
│       └── modelar.py
│
├── docs/
│   └── validacao_manual/
│       └── guia_validacao.csv
│
├── data/ (não versionado)
├── reports/ (não versionado)
│
├── README.md
└── requirements.txt
```

Os diretórios `data/` e `reports/` não são versionados, pois seus conteúdos são gerados dinamicamente a partir dos scripts do projeto.

---

## Pipeline

O fluxo do projeto é organizado em quatro etapas principais.

Inicialmente, os dados são obtidos diretamente da CVM por meio do script de download. Em seguida, ocorre o tratamento e a construção das variáveis, incluindo retornos, medidas de risco, características de tamanho e variáveis de sazonalidade. Após isso, é realizada a modelagem estatística para prever o fluxo futuro. Por fim, é feita uma validação manual para garantir a consistência dos dados em relação aos arquivos brutos.

---

## Execução

Para reproduzir o projeto, basta executar os scripts na seguinte ordem:

```bash
python src/etl/baixar_dados.py
python src/etl/processar_dados.py
python src/model/modelar.py
python src/etl/validar_manual.py
```

---

## Dados

Os dados utilizados são públicos e disponibilizados pela CVM, incluindo:

* Informe diário de fundos de investimento
* Cadastro de fundos

Fonte oficial:

https://dados.cvm.gov.br

Os dados não são armazenados no repositório, sendo sempre obtidos via script, garantindo reprodutibilidade.

---

## Validação Manual

Foi implementado um processo de validação manual com o objetivo de verificar a consistência entre o dataset processado e os dados brutos da CVM.

A validação consiste na seleção aleatória de dois fundos e três datas para cada um. Para essas observações, os valores das variáveis principais são extraídos e organizados em um arquivo de apoio.

Arquivo gerado:

```
docs/validacao_manual/guia_validacao.csv
```

Esse arquivo segue exatamente o formato do informe diário da CVM, contendo as variáveis:

```
TP_FUNDO_CLASSE
CNPJ_FUNDO_CLASSE
ID_SUBCLASSE
DT_COMPTC
VL_TOTAL
VL_QUOTA
VL_PATRIM_LIQ
CAPTC_DIA
RESG_DIA
NR_COTST
```

A partir dele, é possível realizar a verificação diretamente nos arquivos brutos, utilizando filtros simples no Excel.

---

## Modelagem

A modelagem foi estruturada como um problema de regressão com validação temporal, preservando a ordem cronológica dos dados e evitando o uso de embaralhamento aleatório.

As variáveis explicativas incluem retornos passados, medidas de volatilidade, drawdown, tamanho do fundo e indicadores de sazonalidade. O objetivo não é apenas prever o fluxo futuro, mas identificar quais fatores estão mais associados à dinâmica de captação.

---

## Métricas

O desempenho do modelo é avaliado utilizando métricas de erro, como MAE e RMSE, que permitem medir a precisão das previsões em termos absolutos.

---

## Observações

Diferenças pontuais entre os dados processados e os dados brutos podem ocorrer devido a questões de precisão numérica e à existência de múltiplas observações por fundo em uma mesma data. Ainda assim, a validação manual demonstra que o pipeline preserva adequadamente a estrutura dos dados originais.

---

## Reprodutibilidade

O projeto foi estruturado de forma que todas as etapas possam ser reproduzidas a partir dos scripts disponíveis, sem dependência de arquivos externos pré-processados.

---

## Autor

Projeto desenvolvido como parte de um desafio de análise de dados financeiros.
