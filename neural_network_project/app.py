import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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

    def predict(self, X):
        return self.feedforward(X)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y)

st.title("🧠 XOR Neural Network Demo")
st.write("This neural network was trained to solve the XOR problem.")

predictions = nn.predict(X)
st.subheader("Predictions")
for i in range(len(X)):
    st.write(f"Input: {X[i]} → Output: {predictions[i][0]:.3f}")

st.subheader("Loss Curve")
fig, ax = plt.subplots()
ax.plot(nn.losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss over Epochs")
st.pyplot(fig)
