import numpy as np
import pandas as pd
from typing import List, Tuple, override

from src.classifiers import ProbabilisticClassifierModel

class EnsembleClassifier(ProbabilisticClassifierModel):
    """
    Ensemble classifier that combines the predictions of multiple probabilistic models
    by averaging their predicted probability distributions.
    """
    def __init__(self, models: List[ProbabilisticClassifierModel], name: str = "Ensemble"):
        self.models = models
        self.name = name

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits all the underlying models. 
        Note: If models are already fitted, this will re-fit them.
        """
        for model in self.models:
            print(f"Training {model.get_name()}...")
            model.fit(X, y)

    @override
    def predict_proba_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts probabilities by averaging the probability outputs of all base models.
        Returns the mean probabilities and the calculated entropy (uncertainty).
        """
        all_probs = []
        for model in self.models:
            probs, _ = model.predict_proba_with_uncertainty(X)
            all_probs.append(probs)
            
        all_probs = np.array(all_probs)
        mean_probs = np.mean(all_probs, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        
        return mean_probs, entropy

    @override
    def get_name(self) -> str:
        model_names = "+".join([m.get_name() for m in self.models])
        return f"{self.name}({model_names})"

from src.models import ProbabilisticModel

class EnsembleRegressor(ProbabilisticModel):
    """
    Ensemble regressor that combines the predictions of multiple probabilistic models.
    It returns the combined mean and combined standard deviation (uncertainty).
    """
    def __init__(self, models: List[ProbabilisticModel], name: str = "EnsembleReg"):
        self.models = models
        self.name = name

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits all the underlying models. 
        Note: If models are already fitted, this will re-fit them.
        """
        for model in self.models:
            print(f"    [Ensemble] Training {model.get_name()}...")
            model.fit(X, y)

    @override
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        all_means = []
        all_vars = []
        
        for model in self.models:
            mean, std = model.predict_with_uncertainty(X)
            all_means.append(mean)
            all_vars.append(std**2)
            
        all_means = np.array(all_means)
        all_vars = np.array(all_vars)
        
        ens_mean = np.mean(all_means, axis=0)
        ens_var = np.mean(all_vars + all_means**2, axis=0) - ens_mean**2
        ens_var = np.maximum(ens_var, 0.0)
        
        ens_std = np.sqrt(ens_var)
        
        return ens_mean, ens_std

    @override
    def get_name(self) -> str:
        model_names = "+".join([m.get_name().split('(')[0] for m in self.models])
        return f"{self.name}({model_names})"
