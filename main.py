"""Rede neural 4 → 8 → 3 implementada manualmente com NumPy."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
EPOCHS = 1000
LEARNING_RATE = 0.01


def prepare_data():
    """Separa os dados antes de ajustar a padronização, evitando vazamento."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=SEED,
        stratify=iris.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # Cada linha da identidade representa uma classe em one-hot.
    y_train_one_hot = np.eye(3)[y_train]
    y_test_one_hot = np.eye(3)[y_test]
    return x_train, x_test, y_train_one_hot, y_test_one_hot


def initialize_parameters():
    rng = np.random.default_rng(SEED)
    # Escala de He para a camada ReLU; vieses também aleatórios, como pedido.
    return {
        "W1": rng.normal(0, np.sqrt(2 / 4), (4, 8)),
        "b1": rng.normal(0, 0.01, (1, 8)),
        "W2": rng.normal(0, np.sqrt(1 / 8), (8, 3)),
        "b2": rng.normal(0, 0.01, (1, 3)),
    }


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    # Adotamos derivada zero também no ponto z = 0.
    return (z > 0).astype(float)


def softmax(z):
    # Subtrair o maior logit evita overflow sem alterar as probabilidades.
    exponentials = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def forward(x, parameters):
    z1 = x @ parameters["W1"] + parameters["b1"]
    a1 = relu(z1)
    z2 = a1 @ parameters["W2"] + parameters["b2"]
    probabilities = softmax(z2)
    return probabilities, (x, z1, a1)


def cross_entropy(targets, probabilities):
    safe_probabilities = np.clip(probabilities, 1e-15, 1.0)
    return -np.mean(np.sum(targets * np.log(safe_probabilities), axis=1))


def backward(targets, probabilities, cache, parameters):
    x, z1, a1 = cache
    # Softmax + entropia cruzada: dL/dZ2 = (probabilidade - alvo) / N.
    dz2 = (probabilities - targets) / x.shape[0]
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    # Regra da cadeia: propaga o erro pelos pesos e pela derivada da ReLU.
    dz1 = (dz2 @ parameters["W2"].T) * relu_derivative(z1)
    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}


def update_parameters(parameters, gradients, learning_rate):
    for name in parameters:
        parameters[name] -= learning_rate * gradients[name]


def train(x, targets, parameters, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """Gradiente descendente com todo o conjunto de treino a cada atualização."""
    losses = []
    # Época 0: estado inicial. Época 1000: após 1000 atualizações.
    for epoch in range(epochs + 1):
        probabilities, cache = forward(x, parameters)
        loss = cross_entropy(targets, probabilities)
        losses.append(float(loss))
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Época {epoch:4d} | Perda: {loss:.6f}")
        if epoch < epochs:
            gradients = backward(targets, probabilities, cache, parameters)
            update_parameters(parameters, gradients, learning_rate)
    return losses


def evaluate(x, targets, parameters):
    probabilities, _ = forward(x, parameters)
    predictions = np.argmax(probabilities, axis=1)
    actual = np.argmax(targets, axis=1)
    accuracy = np.mean(predictions == actual)
    return float(accuracy), predictions


def main():
    x_train, x_test, y_train, y_test = prepare_data()
    parameters = initialize_parameters()
    print(f"Treinamento: {len(x_train)} amostras | Teste: {len(x_test)} amostras")
    train(x_train, y_train, parameters)
    accuracy, _ = evaluate(x_test, y_test, parameters)
    print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
