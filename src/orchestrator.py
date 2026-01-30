from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.generators import BaseDataGenerator
from src.shadows import BaseShadow
from src.splitter import GapSplitter, SplitResult
from src.models import BaseModel
from src.noise import NoiseStrategy

@dataclass
class ExperimentMetrics:
    mse_train: float
    mse_test_visible: float
    mse_test_shadow: float
    mae_test_shadow: float

@dataclass
class ExperimentArtifacts:
    split_data: SplitResult
    model: BaseModel
    predictions: dict[str, np.ndarray] 
    metrics: ExperimentMetrics

class ExperimentOrchestrator:
    def __init__(self, generator: BaseDataGenerator, splitter: GapSplitter, model: BaseModel):
        self.generator = generator
        self.splitter = splitter
        self.model = model

    def run(self, 
            n_samples: int, 
            x_range: tuple, 
            noise_strategy: NoiseStrategy, 
            shadow: BaseShadow) -> ExperimentArtifacts:
        
        df = self.generator.generate(n_samples, x_range, noise_strategy)
        split_result = self.splitter.split(df, shadow)
        self.model.fit(split_result.X_train, split_result.y_train)
        
        pred_train = self.model.predict(split_result.X_train)
        pred_vis = self.model.predict(split_result.X_test_visible)
        pred_shadow = self.model.predict(split_result.X_test_shadow)
        metrics = ExperimentMetrics(
            mse_train=mean_squared_error(split_result.y_train, pred_train),
            mse_test_visible=mean_squared_error(split_result.y_test_visible, pred_vis),
            mse_test_shadow=mean_squared_error(split_result.y_test_shadow, pred_shadow),
            mae_test_shadow=mean_absolute_error(split_result.y_test_shadow, pred_shadow)
        )

        predictions = {
            'train': pred_train,
            'visible': pred_vis,
            'shadow': pred_shadow
        }

        return ExperimentArtifacts(
            split_data=split_result,
            model=self.model,
            predictions=predictions,
            metrics=metrics
        )
