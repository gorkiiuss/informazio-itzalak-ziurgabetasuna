from typing import override
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier

from src.models import BaseModel, ProbabilisticModel

class ProbabilisticClassifierModel(BaseModel, ABC):
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    @abstractmethod
    def predict_proba_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean_probs, _ = self.predict_proba_with_uncertainty(X)
        return np.argmax(mean_probs, axis=1)

class RandomForestClassifierModel(ProbabilisticClassifierModel):
    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, random_state: int = 42, num_classes: int = 10):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.num_classes = num_classes
        self.name = f"RFClassifier(n={n_estimators}, d={max_depth})"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        if hasattr(self.model, 'classes_'):
            self.num_classes = max(self.num_classes, int(np.max(self.model.classes_)) + 1)
    
    @override
    def predict_proba_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_input = X.values if hasattr(X, 'values') else X

        per_tree_probs = [tree.predict_proba(X_input) for tree in self.model.estimators_]
        
        n_samples = X_input.shape[0]
        n_classes = getattr(self, 'num_classes', 10)
        
        full_per_tree_probs = []
        for probs in per_tree_probs:
            full_probs = np.zeros((n_samples, n_classes))
            for i, cls in enumerate(self.model.classes_):
                full_probs[:, int(cls)] = probs[:, i]
            full_per_tree_probs.append(full_probs)
            
        full_per_tree_probs = np.array(full_per_tree_probs)
        
        mean_probs = np.mean(full_per_tree_probs, axis=0)
        std_probs = np.std(full_per_tree_probs, axis=0)

        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        return mean_probs, entropy
    
    def get_name(self) -> str:
        return self.name

class MCDropoutClassifierNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.out = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

class NeuralNetworkMCDropoutClassifier(ProbabilisticClassifierModel):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10, dropout_rate: float = 0.3, 
                 lr: float = 0.001, epochs: int = 20, batch_size: int = 64, n_iter: int = 50):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_iter = n_iter 

        self.name = f"MCDropoutClass(h={hidden_dim}, p={dropout_rate})"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_tensor = torch.tensor(y_val, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X_tensor.shape[1]
        
        self.network = MCDropoutClassifierNet(input_dim, self.hidden_dim, self.num_classes, self.dropout_rate).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    @override
    def predict_proba_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        import torch.nn.functional as F
        X_val = X.values if hasattr(X, 'values') else X
        
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.network.train()

        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iter):
                logits = self.network(X_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        mean_probs = np.mean(predictions, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        return mean_probs, entropy

    @override
    def get_name(self) -> str:
        return self.name


class BNNClassifierNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        from src.models import BayesianLinear
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.out = BayesianLinear(hidden_dim, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x, sample=True):
        x = self.activation(self.fc1(x, sample))
        x = self.activation(self.fc2(x, sample))
        return self.out(x, sample)
        
    def get_kl(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.out.kl_divergence()

class BayesianNeuralNetworkClassifier(ProbabilisticClassifierModel):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10, lr: float = 0.001, 
                 epochs: int = 20, batch_size: int = 64, n_iter: int = 50, kl_weight: float = 0.001):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_iter = n_iter 
        self.kl_weight = kl_weight

        self.name = f"BNNClass(h={hidden_dim})"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        import torch.nn.functional as F
        global F
        
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_tensor = torch.tensor(y_val, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X_tensor.shape[1]
        
        self.network = BNNClassifierNet(input_dim, self.hidden_dim, self.num_classes).to(self.device)
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.network(batch_X, sample=True)
                nll_loss = F.cross_entropy(outputs, batch_y, reduction='sum')
                kl_loss = self.network.get_kl()
                
                loss = nll_loss + self.kl_weight * kl_loss
                loss.backward()
                optimizer.step()

    @override
    def predict_proba_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        import torch.nn.functional as F
        X_val = X.values if hasattr(X, 'values') else X
        
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.network.eval()

        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iter):
                logits = self.network(X_tensor, sample=True)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        mean_probs = np.mean(predictions, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        return mean_probs, entropy

    @override
    def get_name(self) -> str:
        return self.name

class CNNMCDropoutNet(nn.Module):
    def __init__(self, image_shape: tuple, num_classes: int, dropout_rate: float):
        super().__init__()
        self.image_shape = image_shape
        channels, height, width = image_shape
        
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        final_h = height // 8
        final_w = width // 8
        flattened_size = 128 * final_h * final_w
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.view(batch_size, 32, 32, 3)
        x = x.permute(0, 3, 1, 2)
        
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.reshape(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNMCDropoutClassifier(ProbabilisticClassifierModel):
    def __init__(self, image_shape: tuple = (3, 32, 32), num_classes: int = 10, dropout_rate: float = 0.3, 
                 lr: float = 0.001, epochs: int = 20, batch_size: int = 64, n_iter: int = 50):
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_iter = n_iter 

        self.name = f"CNN+MCDropout(p={dropout_rate})"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_tensor = torch.tensor(y_val, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.network = CNNMCDropoutNet(self.image_shape, self.num_classes, self.dropout_rate).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    @override
    def predict_proba_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        import torch.nn.functional as F
        X_val = X.values if hasattr(X, 'values') else X
        
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        self.network.train() 

        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iter):
                logits = self.network(X_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        mean_probs = np.mean(predictions, axis=0)
        
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        return mean_probs, entropy

    @override
    def get_name(self) -> str:
        return self.name