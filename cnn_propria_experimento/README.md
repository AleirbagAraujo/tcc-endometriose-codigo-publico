# Experimento CNN propria

Esta pasta contem a versao corrigida e reproduzivel da CNN propria. A rede usa
blocos residuais para extrair caracteristicas locais e pooling em multiplas
escalas para incorporar contexto visual.

## Correcoes desta versao

- Ordem fixa das classes: `0 = no_endometriosis` e `1 = endometriosis`.
- Endometriose e sempre a classe positiva das metricas.
- Remocao de imagens exatamente duplicadas antes da divisao.
- Divisao global por grupos de 25 indices consecutivos, mantendo cada grupo
  inteiro e buscando equilibrio entre as classes.
- Operacoes deterministicas, sementes fixas e versoes do ambiente registradas.
- Historico, configuracao, predicoes e metricas salvos em arquivos.
- Figuras geradas sem depender da memoria da sessao do Colab.
- Tamanho de entrada lido automaticamente do modelo na avaliacao e no executor.

O agrupamento por nome e indice e uma aproximacao, nao uma garantia de separacao
por paciente. Caso existam identificadores de paciente, cirurgia ou video, a
divisao deve obrigatoriamente ser refeita por esses identificadores antes da
comparacao final do TCC. Os arquivos atuais nao fornecem essa informacao.

O parametro `--overwrite` somente remove destinos localizados dentro de
`cnn_propria_experimento/resultados`, evitando a exclusao acidental de outras
pastas do Drive.

## Arquitetura

A rede recebe imagens RGB de `224 x 224` pixels. A extracao de caracteristicas
combina uma convolucao inicial com 32 filtros, blocos residuais com 64, 128 e
256 filtros e um modulo multiescala com pooling `5 x 5`, `9 x 9` e `13 x 13`.
Depois da concatenacao, uma convolucao `1 x 1`, global average pooling, dropout
de 0,50 e uma saida sigmoide completam a classificacao. A arquitetura possui
1.474.017 parametros.

## Resultados agregados da execucao de referencia

Antes da divisao, foram auditadas 1.022 imagens. Doze duplicatas exatas foram
removidas, restando 1.010 imagens: 710 para treinamento, 150 para validacao e
150 para teste. Validacao e teste possuem 75 imagens de cada classe.

O treinamento executou 25 epocas e restaurou o modelo da 15a epoca, que obteve
a maior AUC de validacao (`0,9714`). A partir desse ponto, as metricas de treino
continuaram melhorando enquanto a perda de validacao aumentou, indicando
inicio de sobreajuste. O checkpoint selecionado por `val_auc` evitou o uso das
epocas posteriores na avaliacao final.

| Metrica | Resultado no teste |
| --- | ---: |
| Acuracia | 93,33% |
| Precisao | 98,51% |
| Sensibilidade | 88,00% |
| Especificidade | 98,67% |
| F1-score | 92,96% |
| AUC | 0,9751 |

A matriz de confusao apresentou 66 verdadeiros positivos, 74 verdadeiros
negativos, 1 falso positivo e 9 falsos negativos. A elevada precisao e
especificidade mostram poucos alarmes falsos, enquanto a sensibilidade indica
que nove casos positivos nao foram identificados no limiar de 0,50.

## Ordem de execucao no Colab

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/TCC2
```

```bash
!python -m pip install -q -r "cnn_propria_experimento/requirements.txt"
!python "cnn_propria_experimento/00_auditar_imagens.py"
!python "cnn_propria_experimento/01_preparar_dataset.py" --overwrite
!python "cnn_propria_experimento/02_treinar_cnn_propria.py"
!python "cnn_propria_experimento/03_avaliar_cnn_propria.py"
!python "cnn_propria_experimento/04_gerar_figuras_tcc.py"
```

## Estrutura das saidas

```text
cnn_propria_experimento/resultados/
|-- auditoria_imagens.json
|-- dataset_cnn_propria/
|   |-- train/
|   |-- val/
|   |-- test/
|   |-- split_manifest.csv
|   |-- split_summary.csv
|   |-- duplicates_removed.csv
|   `-- split_config.json
|-- resultados_tcc/
|   |-- best_cnn_propria.keras
|   |-- last_cnn_propria.keras (estado da ultima epoca executada)
|   |-- history.csv
|   |-- config_treinamento.json
|   |-- class_names.json
|   |-- model_summary.txt
|   |-- predictions_test.csv
|   |-- test_metrics.json
|   |-- confusion_matrix.png
|   `-- roc_curve.png
`-- figuras_tcc/
    |-- figura_fluxo_cnn_propria.png
    |-- figura_distribuicao_subconjuntos_cnn_propria.png
    |-- figura_curvas_treinamento_cnn_propria.png
    |-- figura_matriz_confusao_cnn_propria.png
    |-- figura_curva_roc_cnn_propria.png
    `-- figura_exemplos_erros_cnn_propria.png
```

Os pesos, imagens, previsoes individuais e demais arquivos de `resultados` nao
fazem parte do repositorio publico.
