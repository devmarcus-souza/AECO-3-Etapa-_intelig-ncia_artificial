# AECO 3 etapa - Inteligência Artificial

Projeto acadêmico para classificar as três espécies do Iris: setosa, versicolor
e virginica. Todos os cálculos da rede e a avaliação são implementados com
NumPy. Scikit-Learn é usado exclusivamente para carregar, dividir e padronizar
os dados; não há modelo ou treinamento pronto.

## Instalação e execução

Abra um terminal nesta pasta e use Python 3. Crie um ambiente independente
para manter as dependências separadas das de outros projetos:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Em Linux/macOS, use `python3 -m venv .venv` e substitua o executável acima por
`.venv/bin/python`.

## Dados e arquitetura

São 150 amostras com quatro medidas de sépalas e pétalas. A divisão
estratificada com semente 42 reserva 120 amostras para treino e 30 para teste.
O `StandardScaler` aprende média e desvio padrão somente no treino. Os rótulos
são convertidos para vetores one-hot com três posições.

| Parâmetro | Dimensão | Função |
|---|---|---|
| W1 | 4 × 8 | Entrada para camada escondida |
| b1 | 1 × 8 | Vieses da camada escondida |
| W2 | 8 × 3 | Camada escondida para saída |
| b2 | 1 × 3 | Vieses da saída |

A camada escondida usa ReLU, `max(0, z)`, com derivada 1 para valores
positivos e 0 para os demais. A saída usa Softmax com subtração do maior
logit de cada amostra para estabilidade numérica. Pesos e vieses são
inicializados aleatoriamente com semente fixa.

## Treinamento e avaliação

O forward pass calcula `Z1 = X @ W1 + b1`, `A1 = ReLU(Z1)` e
`P = Softmax(A1 @ W2 + b2)`. A entropia cruzada é a média de
`-sum(Y * log(P))`, com proteção contra logaritmo de zero.

No backpropagation, o gradiente da saída é `(P - Y) / N`. A regra da cadeia
propaga esse erro para a camada escondida usando a derivada da ReLU.
Todos os parâmetros são atualizados por `parâmetro -= taxa * gradiente`,
usando o conjunto de treino inteiro em cada época.

As constantes `EPOCHS = 1000` e `LEARNING_RATE = 0.01` no início de `main.py`
permitem ajustar o treinamento. A perda é exibida antes das atualizações
(época 0), a cada 100 épocas e após a última atualização.

A classe prevista é o índice da maior probabilidade da Softmax. A acurácia
é calculada manualmente com NumPy: `mean(previsões == classes_reais) * 100`.
O conjunto de teste é usado apenas na avaliação final, sem ajustar parâmetros.
Esse conjunto pequeno fornece uma avaliação acadêmica, não uma garantia de
generalização. Execute o programa para obter a perda e a acurácia locais.

## Resultado verificado

Na execução com os parâmetros padrão, NumPy 2.5.4 e Scikit-Learn 1.9.1,
a perda caiu de **1,899856** para **0,253126**. A acurácia de teste foi
**86,67% (26 acertos em 30 amostras)**. Foram conferidas as dimensões,
a padronização do treino, as somas das probabilidades e o cálculo da
acurácia. Os gradientes dos quatro parâmetros também foram comparados
com aproximações por diferenças finitas e passaram na verificação.
