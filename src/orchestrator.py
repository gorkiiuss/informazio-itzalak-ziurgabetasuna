# src/orchestrator.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss

from src.generators import BaseDataGenerator
from src.shadows import BaseShadow
from src.splitter import GapSplitter, SplitResult
from src.models import BaseModel, DeterministicModel, ProbabilisticModel
from src.classifiers import ProbabilisticClassifierModel
from src.noise import NoiseStrategy

@dataclass
class RegressionMetrics:
    mse_train: float
    mse_test_visible: float
    mse_test_shadow: float
    mae_test_shadow: float

@dataclass
class ClassificationMetrics:
    acc_train: float
    acc_test_visible: float
    acc_test_shadow: float
    log_loss_shadow: float

@dataclass
class BaseArtifacts:
    split_data: SplitResult
    model: BaseModel
    predictions: dict[str, np.ndarray]
    uncertainties: dict[str, np.ndarray] | None

@dataclass
class RegressionArtifacts(BaseArtifacts):
    metrics: RegressionMetrics

@dataclass
class ClassificationArtifacts(BaseArtifacts):
    metrics: ClassificationMetrics

M = TypeVar('M', bound=BaseModel)
T_Metrics = TypeVar('T_Metrics')
T_Artifacts = TypeVar('T_Artifacts', bound=BaseArtifacts)

class BaseOrchestrator(ABC, Generic[M, T_Metrics, T_Artifacts]):
    """
    Edozein esperimenturen oinarrizko fluxua definitzen duen orkestratzaile nagusia:
    Sortu -> Banatu -> Entrenatu -> Iragarri -> Metrikak Kalkulatu
    """
    def __init__(self, generator: BaseDataGenerator, splitter: GapSplitter, model: M):
        self.generator = generator
        self.splitter = splitter
        self.model = model

    def _process_split(self, split_result: SplitResult) -> T_Artifacts:
        self.model.fit(split_result.X_train, split_result.y_train)
        predictions, uncertainties = self._make_predictions(split_result)
        metrics = self._calculate_metrics(split_result, predictions, uncertainties)

        return self._create_artifacts(split_result, predictions, metrics, uncertainties)

    def run(self, 
            n_samples: int, 
            x_range: tuple, 
            noise_strategy: NoiseStrategy, 
            shadow: BaseShadow) -> T_Artifacts:
        
        df = self.generator.generate(n_samples, x_range, noise_strategy)
        split_result = self.splitter.split(df, shadow)
        
        if isinstance(split_result, list):
            split_result = split_result[0]
            
        return self._process_split(split_result)

    def run_cv(self, 
               n_samples: int, 
               x_range: tuple, 
               noise_strategy: NoiseStrategy, 
               shadow: BaseShadow) -> list[T_Artifacts]:
        
        df = self.generator.generate(n_samples, x_range, noise_strategy)
        splits = self.splitter.split(df, shadow)
        
        if not isinstance(splits, list):
            splits = [splits]
            
        return [self._process_split(s) for s in splits]

    @abstractmethod
    def _make_predictions(self, split_result: SplitResult) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        pass

    @abstractmethod
    def _calculate_metrics(self, split_result: SplitResult, predictions: dict[str, np.ndarray], uncertainties: dict[str, np.ndarray] | None) -> T_Metrics:
        pass

    @abstractmethod
    def _create_artifacts(self, split_result: SplitResult, predictions: dict[str, np.ndarray], metrics: T_Metrics, uncertainties: dict[str, np.ndarray] | None) -> T_Artifacts:
        pass

class RegressionOrchestator(BaseOrchestrator[BaseModel, RegressionMetrics, RegressionArtifacts]):
    def _make_predictions(self, split_result: SplitResult) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        uncertainties = None
        
        if isinstance(self.model, ProbabilisticModel):
            pred_train, std_train = self.model.predict_with_uncertainty(split_result.X_train)
            pred_vis, std_vis = self.model.predict_with_uncertainty(split_result.X_test_visible)
            pred_shadow, std_shadow = self.model.predict_with_uncertainty(split_result.X_test_shadow)

            uncertainties = {
                'train': std_train,
                'visible': std_vis,
                'shadow': std_shadow
            }
        else:
            pred_train = self.model.predict(split_result.X_train)
            pred_vis = self.model.predict(split_result.X_test_visible)
            pred_shadow = self.model.predict(split_result.X_test_shadow)

        predictions = {
            'train': pred_train,
            'visible': pred_vis,
            'shadow': pred_shadow
        }
        return predictions, uncertainties

    def _calculate_metrics(self, split_result: SplitResult, predictions: dict[str, np.ndarray], uncertainties: dict[str, np.ndarray] | None) -> RegressionMetrics:
        return RegressionMetrics(
            mse_train=float(mean_squared_error(split_result.y_train, predictions['train'])),
            mse_test_visible=float(mean_squared_error(split_result.y_test_visible, predictions['visible'])),
            mse_test_shadow=float(mean_squared_error(split_result.y_test_shadow, predictions['shadow'])),
            mae_test_shadow=float(mean_absolute_error(split_result.y_test_shadow, predictions['shadow']))
        )

    def _create_artifacts(self, split_result, predictions, metrics, uncertainties):
        return RegressionArtifacts(
            split_data=split_result,
            model=self.model,
            predictions=predictions,
            metrics=metrics,
            uncertainties=uncertainties
        )

class ClassificationOrchestrator(BaseOrchestrator[ProbabilisticClassifierModel, ClassificationMetrics, ClassificationArtifacts]):
    
    def _make_predictions(self, split_result: SplitResult) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        pred_train_prob, std_train = self.model.predict_proba_with_uncertainty(split_result.X_train)
        pred_vis_prob, std_vis = self.model.predict_proba_with_uncertainty(split_result.X_test_visible)
        pred_shadow_prob, std_shadow = self.model.predict_proba_with_uncertainty(split_result.X_test_shadow)

        predictions = {
            'train_prob': pred_train_prob,
            'visible_prob': pred_vis_prob,
            'shadow_prob': pred_shadow_prob,
            'train': np.argmax(pred_train_prob, axis=1),
            'visible': np.argmax(pred_vis_prob, axis=1),
            'shadow': np.argmax(pred_shadow_prob, axis=1)
        }

        uncertainties = {
            'train': std_train,
            'visible': std_vis,
            'shadow': std_shadow
        }
        return predictions, uncertainties

    def _calculate_metrics(self, split_result: SplitResult, predictions: dict[str, np.ndarray], uncertainties: dict[str, np.ndarray] | None) -> ClassificationMetrics:
        num_classes = self.model.num_classes if hasattr(self.model, 'num_classes') else 10
        labels = np.arange(num_classes)

        return ClassificationMetrics(
            acc_train=float(accuracy_score(split_result.y_train, predictions['train'])),
            acc_test_visible=float(accuracy_score(split_result.y_test_visible, predictions['visible'])),
            acc_test_shadow=float(accuracy_score(split_result.y_test_shadow, predictions['shadow'])),
            log_loss_shadow=float(log_loss(split_result.y_test_shadow, predictions['shadow_prob'], labels=labels))
        )

    def _create_artifacts(self, split_result, predictions, metrics, uncertainties):
        final_predictions = {
            'train': predictions['train'],
            'visible': predictions['visible'],
            'shadow': predictions['shadow']
        }
        
        return ClassificationArtifacts(
            split_data=split_result,
            model=self.model,
            predictions=final_predictions,
            metrics=metrics,
            uncertainties=uncertainties
        )