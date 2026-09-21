# src/generators.py

import os
from scipy.io import arff
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

class MNISTGenerator(BaseDataGenerator):
    def __init__(self, data_dir: str = './data', train: bool = True):
        self.data_dir = data_dir
        self.train = train

    @override
    def generate(self, n_samples: int = -1, x_range: tuple[float, float] = (0, 0), noise_strategy: NoiseStrategy | None = None) -> pd.DataFrame:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(self.data_dir, train=self.train, download=True, transform=transform)
        
        X_all = dataset.data.numpy()
        y_all = dataset.targets.numpy()
        
        n_tot = X_all.shape[0]
        X_flat = X_all.reshape(n_tot, -1)
        
        if n_samples > 0 and n_samples < n_tot:
             idx = np.random.choice(n_tot, n_samples, replace=False)
             X_flat = X_flat[idx]
             y_all = y_all[idx]

        X_flat = X_flat.astype(np.float32) / 255.0

        df = pd.DataFrame(X_flat, columns=[f'px_{i}' for i in range(X_flat.shape[1])])
        df['y'] = y_all
        
        return df

class CIFAR10Generator(BaseDataGenerator):
    def __init__(self, data_dir: str = './data', train: bool = True):
        self.data_dir = data_dir
        self.train = train

    @override
    def generate(self, n_samples: int = -1, x_range: tuple[float, float] = (0, 0), noise_strategy: NoiseStrategy | None = None) -> pd.DataFrame:
        import torch
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        dataset = datasets.CIFAR10(self.data_dir, train=self.train, download=True, transform=transform)
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
        all_x_list = []
        all_y_list = []
        for batch_x, batch_y in loader:
            all_x_list.append(batch_x.numpy())
            all_y_list.append(batch_y.numpy())
            
        X_all = np.concatenate(all_x_list)
        X_all = np.transpose(X_all, (0, 2, 3, 1)) 
        y_all = np.concatenate(all_y_list)
        n_tot = X_all.shape[0]
        
        X_flat = X_all.reshape(n_tot, -1)
        
        if n_samples > 0 and n_samples < n_tot:
             idx = np.random.choice(n_tot, n_samples, replace=False)
             X_flat = X_flat[idx]
             y_all = y_all[idx]
             
        columns = [f'pixel_{i}' for i in range(X_flat.shape[1])]
        df = pd.DataFrame(X_flat, columns=columns)
        df['y'] = y_all
        
        return df

class ARFFDataGenerator(BaseDataGenerator):
    def __init__(self, dataset_name: str, target_col: str | None = None, data_dir: str = 'data/weka_arff'):
        self.dataset_name = dataset_name
        self.target_col = target_col
        self.data_dir = data_dir

    @override
    def generate(self, n_samples: int = -1, x_range: tuple[float, float] = (0, 0), noise_strategy: NoiseStrategy | None = None) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{self.dataset_name}.arff")
        
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = df[col].str.decode('utf-8')
                except Exception:
                    pass
        
        if self.target_col is None:
            target_col_name = df.columns[-1]
        else:
            target_col_name = self.target_col
            
        y = getattr(df, target_col_name)
        
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            y = y.astype('category').cat.codes
        else:
            y = pd.to_numeric(y, errors='coerce')
        
        X = df.drop(columns=[target_col_name])
        X = pd.get_dummies(X, drop_first=True, dtype=float)
        X.columns = X.columns.astype(str)
        
        df_final = X.copy()
        df_final['y'] = y.values
        
        df_final = df_final.dropna().reset_index(drop=True)
        
        if 0 < n_samples < len(df_final):
             df_final = df_final.sample(n=n_samples, random_state=42).reset_index(drop=True)
             
        return df_final