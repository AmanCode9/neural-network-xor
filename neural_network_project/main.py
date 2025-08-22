import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.losses = []

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backpropagate(self, X, y):
        output_error = (self.a2 - y) * sigmoid_derivative(self.a2)
        d_W2 = np.dot(self.a1.T, output_error)
        d_b2 = np.sum(output_error, axis=0, keepdims=True)

        hidden_error = np.dot(output_error, self.W2.T) * sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, hidden_error)
        d_b1 = np.sum(hidden_error, axis=0, keepdims=True)

        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.feedforward(X)
            loss = np.mean((y - output) ** 2)
            self.losses.append(loss)
            self.backpropagate(X, y)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.feedforward(X)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y)


print("Predictions:")
print(nn.predict(X))


nn.plot_loss()
