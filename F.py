import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import cross_val_score

# Importing wine data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

print("\nThe wine dataset:\n")
print(data.head())

# Splitting label and features
y = data.quality
X = data.drop('quality', axis=1)

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

# Preprocessing: making X in the range of -1 to 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("\nAfter preprocessing: \n")
print(X_train_scaled)
print(X_train_scaled.shape)

# Using Decision Tree Classifier
clf = tree.DecisionTreeClassifier()

# Fitting: Training the ML Algo
clf.fit(X_train, y_train)

# Obtaining the confidence score for Decision Tree
confidence = clf.score(X_test, y_test)
print("\nThe decision tree confidence score:", confidence)

# PyTorch Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train_scaled)
y_train_tensor = torch.Tensor(y_train.values).view(-1, 1)
X_test_tensor = torch.Tensor(scaler.transform(X_test))
y_test_tensor = torch.Tensor(y_test.values).view(-1, 1)

# Define hyperparameters
input_size = X_train.shape[1]
hidden_size = 32
output_size = 1
learning_rate = 0.001
epochs = 50

# Initialize the neural network, loss function, and optimizer
model = NeuralNetwork(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the neural network
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

# Testing the neural network
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    accuracy = accuracy_score(y_test, y_pred_tensor.round())
    print("\nThe neural network accuracy:", accuracy)

# Create a simple GUI
def predict_quality():
    input_data = [float(entry.get()) for entry in input_entries]
    input_tensor = torch.Tensor(scaler.transform([input_data]))
    prediction = model(input_tensor).item()
    result_label.config(text=f"Predicted Quality: {prediction:.2f}")

# GUI setup
root = tk.Tk()
root.title("Wine Quality Predictor")

input_labels = ["Fixed acidity", "Volatile acidity", "Citric acid", "Residual sugar", "Chlorides",
                "Free sulfur dioxide", "Total sulfur dioxide", "Density", "pH", "Sulphates", "Alcohol"]

input_entries = [tk.Entry(root) for _ in range(len(input_labels))]
for i, label in enumerate(input_labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=5, pady=5)
    input_entries[i].grid(row=i, column=1, padx=5, pady=5)

predict_button = tk.Button(root, text="Predict Quality", command=predict_quality)
predict_button.grid(row=len(input_labels), column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="")
result_label.grid(row=len(input_labels) + 1, column=0, columnspan=2)

root.mainloop()
