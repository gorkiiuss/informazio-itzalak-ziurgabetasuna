# src/models.py

from typing import override
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor

import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

class DeterministicModel(BaseModel, ABC):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

class ProbabilisticModel(BaseModel, ABC):
    @abstractmethod
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

class LinearRegressionModel(DeterministicModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Linear Regression"

class PolynomialRegressionModel(DeterministicModel):
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return f"Polynomial (Degree {self.degree})"

class SVRModel(DeterministicModel):
    def __init__(self, kernel: str = 'rbf', C: float = 100.0, epsilon: float = 0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=epsilon))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return f"SVR ({self.kernel}, C={self.C})"

class NeuralNetworkModel(DeterministicModel):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', max_iter=2000, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True, 
                validation_fraction=0.1
            )
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y.values.ravel())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        layers_str = str(self.hidden_layer_sizes).replace(" ", "")
        return f"NeuralNet {layers_str} ({self.activation})"

class DecisionTreeModel(DeterministicModel):
    def __init__(self, max_depth: int | None = None, random_state: int = 42):
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self.name_suffix = f" (Depth {max_depth})" if max_depth else " (Unlimited)"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def get_name(self) -> str:
        return f"Decision Tree{self.name_suffix}"

class RandomForestModel(ProbabilisticModel):
    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, random_state: int = 42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.name = f"RandomForest(n={n_estimators}, d={max_depth})"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
    
    @override
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_input = X.values if hasattr(X, 'values') else X

        per_tree_pred = [tree.predict(X_input) for tree in self.model.estimators_]
        per_tree_pred = np.array(per_tree_pred)
        
        mean = np.mean(per_tree_pred, axis=0)
        std = np.std(per_tree_pred, axis=0)

        return mean, std
    

    def get_name(self) -> str:
        return self.name

class BaggingPolynomialModel(ProbabilisticModel):
    def __init__(self, degree: int = 2, n_estimators: int = 100, random_state: int = 42):
        self.degree = degree
        self.n_estimators = n_estimators
        
        base_estimator = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        
        self.model = BaggingRegressor(
            estimator=base_estimator, 
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = f"BaggingPoly(deg={degree}, n={n_estimators})"

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    @override
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_input = X.values if hasattr(X, 'values') else X
        
        per_estimator_pred = [est.predict(X_input) for est in self.model.estimators_]
        per_estimator_pred = np.array(per_estimator_pred)
        
        mean = np.mean(per_estimator_pred, axis=0)
        std = np.std(per_estimator_pred, axis=0)

        return mean, std

    @override
    def get_name(self) -> str:
        return self.name

class MCDropoutNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

class NeuralNetworkMCDropoutModel(ProbabilisticModel):
    def __init__(self, hidden_dim: int = 100, dropout_rate: float = 0.2, 
                 lr: float = 0.01, epochs: int = 300, n_iter: int = 100):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.n_iter = n_iter 

        self.name = f"NN+Dropout(h={hidden_dim}, p={dropout_rate})"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None
        self.scaler_X = StandardScaler()

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_scaled = self.scaler_X.fit_transform(X_val)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        input_dim = X_tensor.shape[1]
        
        self.network = MCDropoutNetwork(input_dim, self.hidden_dim, self.dropout_rate).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.network(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    @override
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_val = X.values if hasattr(X, 'values') else X
        
        X_scaled = self.scaler_X.transform(X_val)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.network.train()

        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iter):
                preds = self.network(X_tensor)
                predictions.append(preds.cpu().numpy())

        predictions = np.array(predictions).squeeze()

        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return mean, std

    @override
    def get_name(self) -> str:
        return self.name


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mean=0.0, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).normal_(-3, 0.1))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(out_features).normal_(-3, 0.1))
        
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def get_std(self, rho):
        return torch.log1p(torch.exp(rho))

    def sample_weight(self):
        epsilon = torch.randn_like(self.weight_mu)
        return self.weight_mu + self.get_std(self.weight_rho) * epsilon

    def sample_bias(self):
        epsilon = torch.randn_like(self.bias_mu)
        return self.bias_mu + self.get_std(self.bias_rho) * epsilon

    def forward(self, x, sample=True):
        import torch.nn.functional as F
        
        if sample:
            weight = self.sample_weight()
            bias = self.sample_bias()
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
        
    def kl_divergence(self):
        weight_std = self.get_std(self.weight_rho)
        bias_std = self.get_std(self.bias_rho)
        
        kl_weight = -0.5 * torch.sum(1 + 2 * torch.log(weight_std) - (self.weight_mu ** 2 + weight_std ** 2))
        kl_bias = -0.5 * torch.sum(1 + 2 * torch.log(bias_std) - (self.bias_mu ** 2 + bias_std ** 2))
        
        return kl_weight + kl_bias

class BNNNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.out = BayesianLinear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x, sample=True):
        x = self.activation(self.fc1(x, sample))
        x = self.activation(self.fc2(x, sample))
        return self.out(x, sample)
        
    def get_kl(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.out.kl_divergence()

class BayesianNeuralNetworkModel(ProbabilisticModel):
    def __init__(self, hidden_dim: int = 64, lr: float = 0.01, epochs: int = 500, n_iter: int = 100, kl_weight: float = 0.01):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.n_iter = n_iter 
        self.kl_weight = kl_weight

        self.name = f"BNN(h={hidden_dim})"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None
        self.scaler_X = StandardScaler()

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        import torch.nn.functional as F
        global F
        
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_scaled = self.scaler_X.fit_transform(X_val)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        input_dim = X_tensor.shape[1]
        
        self.network = BNNNetwork(input_dim, self.hidden_dim).to(self.device)
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            outputs = self.network(X_tensor, sample=True)
            nll_loss = F.mse_loss(outputs, y_tensor, reduction='sum')
            kl_loss = self.network.get_kl()
            
            loss = nll_loss + self.kl_weight * kl_loss
            loss.backward()
            optimizer.step()

    @override
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_val = X.values if hasattr(X, 'values') else X
        
        X_scaled = self.scaler_X.transform(X_val)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.network.train()

        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iter):
                preds = self.network(X_tensor, sample=True)
                predictions.append(preds.cpu().numpy())

        predictions = np.array(predictions).squeeze()

        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return mean, std

    @override
    def get_name(self) -> str:
        return self.name
