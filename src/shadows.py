import numpy as np
from abc import ABC, abstractmethod

class BaseShadow(ABC):
    @abstractmethod
    def is_in_shadow(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

class CircularShadow(BaseShadow):
    def __init__(self, x_0: float, y_0: float, r: float):
        self.x_0 = x_0
        self.y_0 = y_0
        self.r = r

    def is_in_shadow(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dist_sq = (x - self.x_0)**2 + (y - self.y_0)**2
        return dist_sq <= self.r**2

class BandShadow(BaseShadow):
    def __init__(self, axis: str, min_val: float, max_val: float):
        if axis.lower() not in ['x', 'y']:
            raise ValueError("The axis must be either 'x' or 'y'.")
            
        self.axis = axis.lower()
        self.min_val = min_val
        self.max_val = max_val

    def is_in_shadow(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        target = x if self.axis == 'x' else y
        return (target >= self.min_val) & (target <= self.max_val)
