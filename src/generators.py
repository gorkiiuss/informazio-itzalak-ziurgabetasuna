from typing import Callable, override
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from src.noise import AbsoluteGaussianNoise, NoiseStrategy

class BaseDataGenerator(ABC):
    @abstractmethod
    def generate(self, n_samples: int, x_range: tuple[float, float], noise_strategy: NoiseStrategy) -> pd.DataFrame:
        pass

class Math2DGenerator(BaseDataGenerator):
    def __init__(self, function: Callable[[np.ndarray], np.ndarray], seed: int = -1):
        self.function: Callable[[np.ndarray], np.ndarray] = function
        seed_rng = None if seed < 0 else seed
        self.rng = np.random.default_rng(seed_rng)

    @override
    def generate(self, n_samples: int = 100, x_range: tuple[float, float] = (0, 10), noise_strategy: NoiseStrategy | None = None) -> pd.DataFrame:
        if noise_strategy is None:
            noise_strategy = AbsoluteGaussianNoise(std_dev=0.1)

        x = self.rng.uniform(x_range[0], x_range[1], n_samples)
        y_pure = self.function(x)
        y = noise_strategy.apply(y_pure)
        df = pd.DataFrame({
            'x': x,
            'y': y
        })

        return df.sort_values(by='x').reset_index(drop=True)
