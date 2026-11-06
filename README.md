# Classificacao de endometriose por visao computacional

Este repositorio apresenta os codigos publicos dos experimentos desenvolvidos
para a classificacao binaria de imagens laparoscopicas nas classes
`endometriosis` e `no_endometriosis`.

O trabalho considera tres arquiteturas: ResNet50, YOLO11n-cls e uma rede
neural convolucional propria. Os codigos dos tres experimentos estao
organizados em pastas independentes e seguem o fluxo numerado de `00` a `04`.

## ResNet50

A ResNet50 foi utilizada por meio de transferencia de aprendizado, com pesos
pre-treinados no conjunto ImageNet. A camada classificadora original foi
substituida por uma cabeca adequada a classificacao binaria. O treinamento foi
realizado em duas etapas: inicialmente com a base convolucional congelada e,
posteriormente, com ajuste fino das camadas finais.

A pasta `resnet50_experimento` contem:

- tratamento das imagens;
- preparacao dos subconjuntos de treinamento, validacao e teste;
- treinamento e ajuste fino da rede;
- avaliacao no conjunto de teste;
- geracao das figuras utilizadas no TCC;
- notebook executor para Google Colab;
- dependencias do experimento.

## YOLO11n-cls

A YOLO11n-cls foi utilizada em sua variante destinada a classificacao de
imagens. O experimento emprega o checkpoint `yolo11n-cls.pt` e realiza o ajuste
do modelo para distinguir imagens com e sem endometriose.

A pasta `yolo11n_cls_experimento` contem:

- conferencia das imagens tratadas;
- preparacao dos subconjuntos de dados;
- treinamento do modelo;
- avaliacao no conjunto de teste;
- geracao das figuras utilizadas no TCC;
- notebook executor para Google Colab;
- dependencias do experimento.

## Rede neural convolucional propria

A terceira abordagem consiste em uma rede neural convolucional desenvolvida
especificamente para o trabalho. Diferentemente da ResNet50 e da YOLO11n-cls,
essa arquitetura nao utiliza pesos pre-treinados: seus parametros sao
inicializados aleatoriamente e aprendidos a partir das imagens do experimento.

A rede recebe imagens RGB com dimensoes de `224 x 224` pixels. Antes da
extracao de caracteristicas, aplica aumento de dados com espelhamento
horizontal, rotacao, zoom e variacao de contraste, seguido da normalizacao dos
valores dos pixels.

A extracao de caracteristicas inicia com 32 filtros e prossegue por blocos
residuais com 64, 128 e 256 filtros. Em seguida, um modulo multiescala aplica
max pooling com janelas `5 x 5`, `9 x 9` e `13 x 13`, concatena essas respostas
e realiza uma projecao `1 x 1`. A classificacao utiliza global average pooling,
dropout de 0,50 e uma unidade de saida sigmoide, que estima a probabilidade de
endometriose.

O treinamento utiliza Adam, entropia cruzada binaria, regularizacao L2 e ate 60
epocas. A parada antecipada e a selecao do melhor modelo usam a AUC de
validacao. A ordem das classes e fixada em `0 = no_endometriosis` e
`1 = endometriosis`, de modo que a endometriose seja sempre a classe positiva.

Na execucao de referencia, o melhor modelo foi obtido na 15a epoca. No conjunto
de teste com 150 imagens, a CNN propria alcancou acuracia de 93,33%, precisao de
98,51%, sensibilidade de 88,00%, especificidade de 98,67%, F1-score de 92,96%
e AUC de 0,9751. Esses valores sao agregados; imagens, pesos e predicoes
individuais nao sao publicados.

## Dados e arquivos nao publicados

Por se tratar de um repositorio publico, nao sao disponibilizados conjuntos de
imagens, pesos treinados, previsoes individuais ou resultados contendo imagens
laparoscopicas.

Os codigos destinam-se exclusivamente a pesquisa academica. As classificacoes
produzidas pelos modelos nao constituem diagnostico medico nem validacao
clinica.
