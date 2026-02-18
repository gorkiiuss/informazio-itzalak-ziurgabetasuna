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

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass

class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Linear Regression"

class PolynomialRegressionModel(BaseModel):
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return f"Polynomial (Degree {self.degree})"

class SVRModel(BaseModel):
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

class NeuralNetworkModel(BaseModel):
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

class DecisionTreeModel(BaseModel):
    def __init__(self, max_depth: int | None = None, random_state: int = 42):
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self.name_suffix = f" (Depth {max_depth})" if max_depth else " (Unlimited)"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def get_name(self) -> str:
        return f"Decision Tree{self.name_suffix}"

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, random_state: int = 42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y.values.ravel())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return "Random Forest"
