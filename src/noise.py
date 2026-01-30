from typing import override
import numpy as np
from abc import ABC, abstractmethod
class NoiseStrategy(ABC):
    def __init__(self, seed: int = -1):
        self.seed = seed
        rng_seed = None if seed < 0 else seed
        self.rng = np.random.default_rng(rng_seed)

    @abstractmethod
    def apply(self, y_pure: np.ndarray) -> np.ndarray:
        pass

class AbsoluteGaussianNoise(NoiseStrategy):
    def __init__(self, std_dev: float = 1.0, seed: int = -1):
        super().__init__(seed)
        self.std_dev = std_dev

    def apply(self, y_pure: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0, self.std_dev, size=y_pure.shape)
        return y_pure + noise

class AdaptiveGaussianNoise(NoiseStrategy):
    def __init__(self, noise_ratio: float = 0.1, min_noise: float = 1e-3, seed: int = -1):
        super().__init__(seed)
        self.noise_ratio = noise_ratio
        self.min_noise = min_noise
   
    def apply(self, y_pure: np.ndarray) -> np.ndarray:
        signal_std = np.std(y_pure)
        
        if signal_std == 0:
            effective_std = self.min_noise
        else:
            effective_std = signal_std * self.noise_ratio
            
        noise = self.rng.normal(0, effective_std, size=y_pure.shape)
        return y_pure + noise
