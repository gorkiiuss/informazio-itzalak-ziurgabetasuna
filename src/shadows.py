# src/shadows.py

import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

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

class ClassShadow(BaseShadow):
    def __init__(self, shadow_class):
        self.shadow_class = shadow_class

    def is_in_shadow(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if isinstance(self.shadow_class, (list, tuple)):
            return np.isin(y, self.shadow_class)
        return y == self.shadow_class

class ClosestClusterShadow(BaseShadow):
    def __init__(self, k_range: tuple[int, int] = (5, 10), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state
        self.best_k = None
        self.best_labels = None
        self.best_centers = None

    def get_cluster_labels(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        best_k = self.k_range[0]
        best_score = -1
        best_labels = None
        best_centers = None
        
        X_arr = np.asarray(x)
        
        max_k = min(self.k_range[1], len(X_arr) - 1)
        
        if max_k < 2:
            return np.zeros(len(X_arr), dtype=int)

        for k in range(self.k_range[0], max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
            labels = kmeans.fit_predict(X_arr)
            score = silhouette_score(X_arr, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_centers = kmeans.cluster_centers_
                
        self.best_k = best_k
        self.best_labels = best_labels
        self.best_centers = best_centers
        return best_labels

    def is_in_shadow(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        best_labels = self.get_cluster_labels(x, y)
        
        if len(np.unique(best_labels)) < 2:
            return np.zeros(len(x), dtype=bool)
                
        dist_matrix = cdist(self.best_centers, self.best_centers)
        np.fill_diagonal(dist_matrix, 0.0)
        sum_dists = np.sum(dist_matrix, axis=1)
        closest_cluster_idx = np.argmin(sum_dists)
        
        return best_labels == closest_cluster_idx

